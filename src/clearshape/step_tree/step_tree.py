# TODO add tree_to_dgl function
# TODO allow parsing of ComplexEntityInstances if possible
#%%
"""
This module provides functionalities for parsing and representing STEP (Standard for the Exchange of Product model data) files as tree structures. It utilizes the `steputils` library for processing STEP files and `numpy` for handling numerical operations related to the tree structure.

This module uses some STEP file specific terminology:
- step entity instance (or step instance for short)
    The concret object specified in one line of a STEP file and refert to by its reference number (e.g. "#123")
    (not to be confused with a "step entity")
- step entity
    Type of the object instanciated in a step file. (e.g. CLOSED_SHELL, CARTESIAN_POINT)
- numeric atomic and non-numeric atomics
    Numeric atomics refer to all characters making up a number (including signs, digits, decimal points). Non-numeric atomic are every thing else (e.g. booleans)


Classes
-------
StepTree
    Represents a tree structure with nodes and edges, designed to model the hierarchical organization of data in a STEP file.

See Also
--------
steputils : A utility library for working with STEP files.
numpy : A fundamental package for scientific computing with Python.
"""

__author__ = "Daniel Mansfeldt"

# standard libaries
from collections.abc import Iterable
import sys
import os
from pathlib import Path

# Third party libaries
from steputils import p21
import numpy as np
import dgl
import torch
import pickle
import matplotlib.pyplot as plt
import networkx as nx

# application imports
from clearshape.step_tree.node import Node
import clearshape.step_tree.step_analysis as sta


class StepTree:
    """
    Represents a tree structure derived from a STEP file, containing nodes and edges to model the hierarchical data.

    Parameters
    ----------
    nodes : list[Node]
        A list of `Node` objects representing the geometric entitiy instances, boolean values and numerber characters in the STEP file.
    edges : list[tuple]
        A list of tuples representing the edges between nodes, where each tuple contains the identifiers of the connected nodes.

    Attributes
    ----------
    nodes : list[Node]
        Stores the nodes of the tree.
    edges : list[tuple]
        Stores the edges between nodes in the tree.

    Examples
    --------
    >>> # create StepTree instance from STEP file and convert it to DGL graph
    >>> tree = StepTree.from_step_file("path.stp")
    >>> tree_as_graph = tree.to_dgl_graph()
    """

    def __init__(self, nodes: list[Node], edges: list[tuple]):
        self.nodes = nodes
        self.edges = edges

    @classmethod
    def from_step_file(cls, path: str):
        """
        Creates a `StepTree` instance from a STEP file located at the specified path.

        This method reads a STEP file, parses its contents to extract nodes and edges representing the hierarchical structure of the STEP data, and then creates a `StepTree` instance with this information.

        Parameters
        ----------
        path : str
            The file system path to the STEP file that is to be read and parsed.

        Returns
        -------
        StepTree
        An instance of the `StepTree` class populated with nodes and edges derived from the STEP file.

        See Also
        --------
        p21.readfile : Function from the `steputils` library used to read STEP files.
        _parse_step_file : A private method of `StepTree` class that parses the raw data from a STEP file into nodes and edges.

        Examples
        --------
        >>> step_tree = StepTree.from_step_file('path/to/your/step_file.stp')
        >>> print(type(step_tree))
        <class '__main__.StepTree'>

        Notes
        -----
        This method assumes the STEP file is well-formed and compliant with the ISO 10303-21 standard. It relies on the `steputils` library for parsing the STEP file, which should be properly installed and configured.
        """
        # TODO assure step file can be read regardless of capitalization of the file suffix
        step_file = p21.readfile(path)

        # check step file to be suitable for conversion into tree
        count_by_entity_name = sta.entity_counts(step_file)
        assert count_by_entity_name["CLOSED_SHELL"] == 1, f'Step file must only contain one CLOSED_SHELL instance, but has {count_by_entity_name["CLOSED_SHELL"]}.'

        nodes, edges = StepTree._parse_step_file(step_file)
        return cls(nodes, edges)
    
    @staticmethod
    def from_pickle_file(path: str):
        """
        Creates an instance of `StepTree` from a previously saved Pickle file.

        This method loads a previously saved Pickle file containing the data of a `StepTree` instance and then creates a corresponding instance of `StepTree`.

        Parameters
        ----------
        path : str
            The filesystem path to the Pickle file to be loaded.

        Returns
        -------
        StepTree
            An instance of the class `StepTree` created from the loaded Pickle file.

        Examples
        --------
        >>> step_tree = StepTree.from_pickle_file('path/to/your/pickle_file.pkl')
        >>> print(type(step_tree))
        <class '__main__.StepTree'>
        """
        with open(path, "rb") as f:
            loaded_data = pickle.load(f)
        
        return StepTree(**loaded_data)
    
    def to_pickle_file(self, path: str):
        """
        Saves the data of the current `StepTree` instance to a Pickle file.

        This method serializes the data of the current `StepTree` instance and saves it to a Pickle file at the specified path.

        Parameters
        ----------
        path : str
            The filesystem path under which the Pickle file should be saved.

        Returns
        -------
        None

        Examples
        --------
        >>> step_tree = StepTree()  # Assuming an instance of StepTree has been created previously.
        >>> step_tree.to_pickle_file('path/to/your/pickle_file.pkl')
        # Data successfully saved to 'path/to/your/pickle_file.pkl'.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def show_graph(self):
        """
        Visualizes the graph represented by the StepTree instance.

        This method creates a visualization of the graph represented by the current StepTree instance using the `networkx` library and `matplotlib`.

        Returns
        -------
        None

        Notes
        -----
        The graph is created using the `dgl` library, which converts the edges of the StepTree instance into a directed graph (`DGLGraph`). This directed graph is then converted into an undirected graph (`networkx.Graph`) to visualize using `networkx` and `matplotlib`.

        Examples
        --------
        >>> step_tree = StepTree(nodes, edges)  # Assuming nodes and edges are defined
        >>> step_tree.show_graph()
        # Displays the visualization of the graph represented by the StepTree instance.
        """
        G = dgl.graph(self.edges)
        nx_G = G.to_networkx().to_undirected()
        nx.draw(nx_G, with_labels=False, node_color='skyblue', node_size=20, font_size=5)
        plt.show()

    @classmethod
    def _parse_step_file(cls, stepfile: p21.StepFile):
        """
        Parses a `p21.StepFile` object to extract nodes and edges suitable for constructing a `StepTree`.

        This method processes a STEP file loaded into a `p21.StepFile` object, identifying and creating nodes representing STEP entities and edges that define their relationships. It initializes the tree with a root node and iteratively processes each entity to build the tree structure.

        Parameters
        ----------
        stepfile : p21.StepFile
            The STEP file object loaded by the `steputils` library, which contains the hierarchical data of the STEP file.

        Returns
        -------
        tuple of (list[Node], list[tuple])
            A tuple containing two lists: the first list comprises `Node` objects representing entities in the STEP file, and the second list comprises tuples representing edges between these nodes.

        Notes
        -----
        The method starts by creating a root node based on the `CLOSED_SHELL` entity (or an equivalent root entity in the STEP file). It then processes the file's entities to extract child nodes and their relationships, represented by edges. The process involves:

        - Mapping numerical references in entities to node IDs.
        - Handling non-numeric atomics, with a potential improvement noted for consolidating boolean values into single nodes.
        - Recursively processing child entities to expand the tree with correct hierarchical relationships.

        This method is a core part of the `StepTree` class's functionality, enabling the conversion of flat STEP file data into a structured tree representation. The detailed processing logic, including handling of specific entity types and their parameters, is crucial for accurately reflecting the STEP file's structure in the tree.
        """
        nodes = []
        edges = []
        unique_numbers_to_start_node_id_map = {}

        # add first node (Root node aka. CLOSED_SHELL)
        root_node_reference = cls._get_instance_reference(stepfile)
        nodes.append(
            Node.from_step_instance(0, stepfile.data[0].instances[root_node_reference])
        )

        nodes_to_process = nodes.copy()  # only one at this point
        while nodes_to_process:
            node_to_be_processed = nodes_to_process[0]
            # get parameters from step entity instances
            child_instances_references, child_atomics, numbers = cls._parse_parameters(
                stepfile.data[0].instances[node_to_be_processed.instance_reference]
            )

            # process numbers
            nodes, edges, unique_numbers_to_start_node_id_map = cls._process_numbers(
                nodes,
                edges,
                numbers,
                unique_numbers_to_start_node_id_map,
                node_to_be_processed,
            )

            # process non-numeric atomics
            # TODO Possible improvement: Keeping only one node for each boolean
            # value and linking them to all nodes referencing to a True or False
            # value.
            nodes, edges = cls._process_non_numeric_atomics(
                nodes, edges, node_to_be_processed, child_atomics
            )

            # process child instances
            nodes, edges, nodes_to_process = StepTree._process_child_instances(
                nodes,
                edges,
                node_to_be_processed,
                nodes_to_process,
                stepfile,
                child_instances_references,
            )

            # remove processed node
            nodes_to_process.pop(0)

        return nodes, edges

    # TODO add test case
    @classmethod
    def _process_numbers(
        cls,
        nodes,
        edges,
        numbers,
        unique_numbers_to_start_node_id_map,
        node_to_be_processed,
    ):
        """
        Processes numerical values from a STEP entity instance, integrating them into the tree structure.

        This method takes a list of numbers extracted efrom a STEP entity and either connects the current node to an existing sequence of nodes representing the same number or creates a new sequence of nodes for this number. It ensures that each unique number in the STEP file is represented by a unique sequence of nodes in the tree, to maintain an efficient and accurate representation of numerical data.

        Parameters
        ----------
        nodes : list[Node]
            The current list of nodes in the tree.
        edges : list[tuple]
            The current list of edges in the tree, where each edge is a tuple of node IDs.
        numbers : list[int] or list[float]
            A list of numerical values to be processed and integrated into the tree.
        unique_numbers_to_start_node_id_map : dict
            A dictionary mapping each unique number to the ID of the starting node of its representing sequence in the tree.
        node_to_be_processed : Node
            The current node being processed, which the numerical values are associated with.

        Returns
        -------
        tuple of (list[Node], list[tuple], dict)
            A tuple containing the updated lists of nodes and edges, and the updated dictionary of unique numbers after processing the input numbers. The dictionary maps unique numbers to the IDs of their corresponding starting nodes in the tree.

        Notes
        -----
        This method checks if a numerical value has already been represented in the tree. If so, it reuses the existing node sequence for that number by creating an edge from the current node to the start of the sequence. If the number has not been represented, it generates a new sequence of nodes for that number, updates the dictionary of unique numbers, and integrates this new sequence into the tree.

        The purpose of maintaining a unique sequence of nodes for each number is to avoid redundancy and ensure that the tree structure efficiently represents numerical data with minimal duplication.
        """
        for number in numbers:
            # check wether to reuse exiting string of nodes representing the
            # exact same number.
            if number in unique_numbers_to_start_node_id_map.keys():
                # attach current node to existing node number string
                edges.append(
                    (
                        node_to_be_processed.id,
                        unique_numbers_to_start_node_id_map[number],
                    )
                )

            else:
                number_string_nodes, number_string_edges = (
                    cls._get_node_string_from_number(number, nodes)
                )

                # add unique number to dict of unique numbers
                unique_numbers_to_start_node_id_map[number] = number_string_nodes[0].id

                # attach node string to parent node
                edges.append((node_to_be_processed.id, number_string_nodes[0].id))
                edges += number_string_edges
                nodes += number_string_nodes
        return nodes, edges, unique_numbers_to_start_node_id_map

    # TODO add test case
    @classmethod
    def _process_non_numeric_atomics(
        cls, nodes, edges, node_to_be_processed, child_atomics
    ):
        """
        Processes non-numeric atomic values, creating new nodes for each and updating the tree structure.

        This method handles the integration of non-numeric atomic values associated with a STEP file entity into the tree; mainly booleans. For each atomic value, it creates a new node and establishes an edge from the currently processed node to this new node.

        Parameters
        ----------
        nodes : list[Node]
            The current list of nodes in the tree.
        edges : list[tuple]
            The current list of edges in the tree, where each edge is a tuple of node IDs.
        node_to_be_processed : Node
            The current node being processed, to which the non-numeric atomic values are associated.
        child_atomics : list
            A list of non-numeric atomic values to be processed and integrated into the tree.

        Returns
        -------
        tuple of (list[Node], list[tuple])
            A tuple containing the updated lists of nodes and edges after integrating the non-numeric atomic values into the tree.
        """
        for atomic in child_atomics:
            new_node_id = cls._get_next_availabel_node_id(nodes)
            new_node = Node.from_atomic(new_node_id, atomic)
            nodes.append(new_node)
            edges.append((node_to_be_processed.id, new_node.id))
        return nodes, edges

    # TODO add test case
    @classmethod
    def _process_child_instances(
        cls,
        nodes,
        edges,
        node_to_be_processed,
        nodes_to_process,
        stepfile,
        child_instances_references,
    ):
        """
        Processes child instance references, creating nodes and establishing parent-child relationships in the tree.

        This method iterates over a list of child instance references extracted from a STEP file entity. It checks if a node already exists for each reference. If so, it creates an edge between the current node and the existing node. If not, it creates a new node for the reference, adds it to the tree, and schedules it for further processing. This approach ensures that the hierarchical structure of the STEP file is accurately represented in the tree.

        Parameters
        ----------
        nodes : list[Node]
            The current list of nodes in the tree.
        edges : list[tuple]
            The current list of edges in the tree, where each edge is a tuple of node IDs.
        node_to_be_processed : Node
            The node currently being processed.
        nodes_to_process : list[Node]
            The list of nodes that are queued for processing. This method may add new nodes to this list.
        stepfile : p21.StepFile
            The STEP file being processed, loaded into a `p21.StepFile` object.
        child_instances_references : list[int]
            A list of instance references for child entities found in the current STEP entity.

        Returns
        -------
        tuple of (list[Node], list[tuple], list[Node])
            A tuple containing the updated lists of nodes and edges, and the updated list of nodes queued for further processing.
        """
        for instance_reference in child_instances_references:
            # check if a node for this step entity instance already exists
            if instance_reference in [node.instance_reference for node in nodes]:
                # add edge between currently processed node and existing
                # node, which refers to an previously processed step
                # instance.
                existing_node = [
                    node
                    for node in nodes
                    if node.instance_reference == instance_reference
                ][0]

            else:
                new_node_id = cls._get_next_availabel_node_id(nodes)
                try:
                    new_node = Node.from_step_instance(
                        new_node_id, stepfile.data[0].instances[instance_reference]
                    )
                    nodes.append(new_node)
                    edges.append((node_to_be_processed.id, new_node.id))
                    nodes_to_process.append(new_node)
                except ValueError:
                    pass  # skip instance not feeding into closed shell
        return nodes, edges, nodes_to_process

    @classmethod
    def _get_instance_reference(
        cls, step_file: p21.StepFile, entity_name: str = "CLOSED_SHELL"
    ) -> str:
        """
        Searches for a specific entity by name within a STEP file and returns its instance reference.

        This method iterates through the instances in a given STEP file, looking for an entity that matches the specified name. It is primarily used to find the starting point for constructing the tree structure by identifying a key entity (e.g., "CLOSED_SHELL") as the root. The method skips over complex entity instances.

        Parameters
        ----------
        step_file : p21.StepFile
            The STEP file being processed, encapsulated in a `p21.StepFile` object.
        entity_name : str, optional
            The name of the entity to find within the STEP file. Defaults to "CLOSED_SHELL".

        Returns
        -------
        str
            The reference of the found entity instance, formatted as a string (e.g., '#100').

        Examples
        --------
        >>> step_file = p21.readfile('path/to/step_file.stp')
        >>> instance_ref = StepTree._get_instance_reference(step_file)
        >>> print(instance_ref)
        '#100'
        """
        for _, instance in step_file.data[0].instances.items():
            if isinstance(instance, p21.ComplexEntityInstance):
                continue
            if instance.entity.name == entity_name:
                return instance.ref

    @classmethod
    def _parameter_gen(cls, parameters):
        """
        A generator function to recursively iterate through nested iterables, yielding each element in a flattened manner.

        This method is designed to handle arbitrarily nested iterables (such as lists or tuples) by recursively traversing them and yielding  a list of the individual elements. It treats strings as individual elements, not iterables, to avoid breaking them down into characters. This functionality is useful for processing data structures that contain nested lists or tuples, especially when the depth of nesting is variable or unknown ahead of time.

        Parameters
        ----------
        parameters : iterable
            The iterable, which may be nested to any level, to be flattened. This can include lists, tuples, and any other objects that are iterable, with the exception of strings, which are treated as individual elements.

        Yields
        ------
        element
            Individual elements from the provided iterable, yielded one by one in a flattened manner. This includes elements from deeply nested structures, presented as if they were in a single, non-nested iterable.

        Examples
        --------
        >>> nested_list = [1, [2, 3], [4, [5, 6]], 'string']
        >>> for element in StepTree._parameter_gen(nested_list):
        ...     print(element)
        1
        2
        3
        4
        5
        6
        'string'
        """
        for parameter in parameters:
            if isinstance(parameter, Iterable) and not isinstance(parameter, str):
                yield from cls._parameter_gen(parameter)
            else:
                yield parameter

    @classmethod
    def _parse_parameters(cls, instance: p21.SimpleEntityInstance):
        """
        Parses the parameters of a STEP file entity instance, categorizing them into child instances, atomic values, and numerical values.

        This method processes the parameters of a given `p21.SimpleEntityInstance`, categorizing the parameters into three groups: references to child instances (indicated by a '#' prefix), atomic values (specifically '*', '.F.', and '.T.'), and numerical values (integers or floats). This categorization is essential for constructing the hierarchical tree structure of the STEP file data, allowing for the differentiation between entity references, boolean values, and numerical data.

        Parameters
        ----------
        instance : p21.SimpleEntityInstance
            An instance of a STEP file entity to be processed. The instance should be a `p21.SimpleEntityInstance`, containing parameters that represent the entity's data.

        Returns
        -------
        tuple of (list[str], list[str], list[Union[int, float]])
            A tuple containing three lists:
            - The first list contains the references to child instances, identified by a '#' prefix.
            - The second list contains atomic values, specifically '*', '.F.', and '.T.', which represent undefined, FALSE, and TRUE, respectively.
            - The third list contains numerical values, including both integers and floats.

        Raises
        ------
        TypeError
            If the provided instance is not a `p21.SimpleEntityInstance`, indicating that the input is invalid for processing.

        Examples
        --------
        >>> instance = p21.SimpleEntityInstance(...)
        >>> child_instances, child_atomics, numbers = StepTree._parse_parameters(instance)
        >>> print(child_instances, child_atomics, numbers)
        ['#123', '#124'], ['*', '.T.'], [1, 2.5]
        """
        if not isinstance(instance, p21.SimpleEntityInstance):
            raise TypeError("Invalid input.")

        parameters_flat = list(cls._parameter_gen(instance.entity.params))

        parameters_string = [
            parameter for parameter in parameters_flat if isinstance(parameter, str)
        ]
        child_instances = [
            parameter for parameter in parameters_string if parameter.startswith("#")
        ]

        child_atomics = [
            parameter
            for parameter in parameters_flat
            if parameter in ["*", ".F.", ".T."]
        ]

        numbers = [
            parameter
            for parameter in parameters_flat
            if isinstance(parameter, (int, float))
        ]
        return child_instances, child_atomics, numbers

    @classmethod
    def _get_node_string_from_number(cls, number: int, nodes: list[Node]):
        """
        Creates a sequence of nodes representing each digit or character in the specified number, along with the edges that connect these nodes.

        This method converts a number into its string representation, then reverses this string to create nodes for each character in the correct order for tree construction. The method ensures that the numerical value is represented in a way that does not include trailing zeros after the decimal point.

        Parameters
        ----------
        number : int
            The number to be converted into a string of nodes within the tree.
        nodes : list[Node]
            The current list of nodes in the tree, used to ensure unique identification for new nodes created from the number.

        Returns
        -------
        tuple of (list[Node], list[tuple])
            A tuple containing two elements:
            - The first element is a list of `Node` objects, each representing a character from the number's string representation.
            - The second element is a list of tuples, each representing an edge between consecutive nodes in the number string, effectively connecting the nodes in a linear sequence.

        Notes
        -----
        The conversion of the number to its string representation uses `numpy.format_float_positional` to avoid scientific notation and ensure no trailing zeros are present after a decimal point.
        """
        # the first digit or sign of the number is supposed to be a leaf in the
        # tree. Therefore the number string is reversed, as this is the correct
        # order to string the resulting nodes together.
        number = np.format_float_positional(number)
        number_as_character_list_reversed = list(reversed(number))

        nodes_from_number = []
        for charcter_index, character in enumerate(number_as_character_list_reversed):

            character_node_id = cls._get_next_availabel_node_id(nodes) + charcter_index
            character_node = Node.from_atomic(character_node_id, character)
            nodes_from_number.append(character_node)

        # build egdes within node string
        edges = []
        for node_index, node in enumerate(nodes_from_number):
            try:
                edges.append((node.id, nodes_from_number[node_index + 1].id))
            except IndexError:
                pass  # last node has no child to form an edge with

        return nodes_from_number, edges

    @staticmethod
    def _get_next_availabel_node_id(nodes: list[Node]) -> int:
        """
        Computes the next available node ID based on the highest current node ID in the list of nodes.

        This method facilitates the assignment of unique IDs to new nodes by identifying the highest node ID in the current list of nodes and incrementing it by one. It ensures that each node within the tree structure has a unique identifier, which is crucial for accurately representing relationships between nodes and for the integrity of the tree's structure.

        Parameters
        ----------
        nodes : list[Node]
            The list of current nodes in the tree.

        Returns
        -------
        int
            The next available node ID, which is one greater than the highest node ID found in the list of current nodes.

        Examples
        --------
        >>> nodes = [Node(1, 'TypeA'), Node(2, 'TypeB'), Node(3, 'TypeC')]
        >>> next_id = StepTree._get_next_available_node_id(nodes)
        >>> print(next_id)
        4

        Notes
        -----
        This method assumes that node IDs are sequential integers and that the list of nodes provided as input includes at least one node. If the list is empty, this method will raise a `ValueError` due to the inability to find the maximum of an empty sequence.
        """
        return max([node.id for node in nodes]) + 1

    # TODO add docstring
    def one_hot_encode_node_labels(self):
        entity_encoding = torch.zeros((len(self.nodes), 32))
        for node in self.nodes:
            entity_encoding[node.id, node.label] = 1
        return entity_encoding

    # TODO add docstring
    # TODO write test
    # TODO add cuda support for dgl graph
    def to_dgl_graph(self):
        """Convert tree to dgl graph.
        Reverses edges so that all edges flow into closed shell.
        """
        dgl_graph = dgl.graph(self.edges).reverse()

        # if torch.cuda.is_available():
        #    dgl_graph = dgl_graph.to("cuda:0")

        dgl_graph.ndata["node_classes"] = self.one_hot_encode_node_labels()

        return dgl_graph

# %%
