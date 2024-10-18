import networkx as nx


def get_next(nodes: list[str] | set[str]):
    """Get next node in list of nodes."""
    next_node = sorted(nodes).pop()
    nodes.remove(next_node)
    return next_node


def iter_depth_first_search(g: nx.DiGraph, logging_func=None):
    """
    Iterate through the network from upstream to downstream.

    References:
        In the paper: See section "Methodology" - subsection "Search algorithm".

    Args:
        g (nx.DiGraph):
        logging_func (function): function used for logging some messages. Default: no messages logged.

    Yields:
        str: label of the next node in the network.
    """
    # node which are already processed
    nodes_done = set()

    # source nodes of the network
    nodes_start = set(n for n in g if g.in_degree[n] == 0)

    # number of nodes in the network
    n_nodes = len(g)

    # current node to be processed
    node_now = get_next(nodes_start)

    # default is no logging output.
    if logging_func is None:
        def logging_func(msg): ...

    i = 1
    while i < n_nodes + 1:
        logging_func(f"--- {node_now = } | {i = } ---")

        if node_now in nodes_done:
            logging_func("ALREADY DONE")

        yield node_now
        i += 1

        nodes_done.add(node_now)

        if g.out_degree[node_now] == 0:
            # yield None
            try:
                logging_func("No node downstream -> pick next start node\n" + "_" * 25)
                node_now = get_next(nodes_start)
            except (KeyError, IndexError):
                logging_func("No more start nodes after final node")
                break
        elif g.out_degree[node_now] == 1:
            logging_func("One node downstream -> next node downstream")
            node_now = next(g.successors(node_now))
            ...
        else:
            # Branching node
            logging_func("Multiple nodes downstream")
            node_next = None
            for node_out in g.successors(node_now):
                if node_out in nodes_done:
                    ...
                elif node_out in nodes_start:
                    ...
                elif node_next is None:
                    node_next = node_out
                    logging_func(f'- Pick one downstream -> "{node_out}"')
                else:
                    if all(
                        (node_in in nodes_done) for node_in in g.predecessors(node_out)
                    ):
                        # all node already processed.
                        logging_func(f'- Add other to start nodes -> "{node_out}"')
                        nodes_start.add(node_out)

                    else:
                        logging_func(f'- skip due to other path -> "{node_out}"')

            node_now = node_next
            del node_next

        # ---
        if g.in_degree[node_now] == 0:
            logging_func("No node upstream")
            # start node -> OK
            ...
        elif g.in_degree[node_now] == 1:
            logging_func("One node upstream")
            # linear -> OK
            ...
        else:
            # Merging node
            logging_func("Multiple nodes upstream -> pick next start node\n" + "_" * 25)
            # search for the next source node (node which is unprocessed)
            if any((node_in not in nodes_done) for node_in in g.predecessors(node_now)):
                # yield None
                try:
                    node_now = get_next(nodes_start)
                except (KeyError, IndexError):
                    logging_func("No more start nodes after join")
                    break
