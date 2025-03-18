def merge_sorted_lists(list1, list2):

    def safe_index(lst, i):
        if (i >= 0 and i < len(lst)) or (i < 0 and i >= -len(lst)):
            return lst[i].lower()
        else:
            return chr(1114111)
    
    # Initialize pointers for both lists
    i, j = 0, 0
    merged_list = []
    list1_inds = []
    list2_inds = []
    
    curr_ind = -1
    # Traverse both lists and insert smaller value from either list into the merged list
    while i < len(list1) or j < len(list2):

        # print(merged_list)
        # print("1", safe_index(list1, i), i)
        # print("2", safe_index(list2, j), j)
        
        # never repeat an entry
        # you're only ever at risk of doing this if the last entry is the same as your current one
        if safe_index(list1, i) == safe_index(merged_list, -1):
            # print("1 repeat")
            list1_inds.append(curr_ind)
            i += 1
        elif safe_index(list2, j) == safe_index(merged_list, -1):
            # print("2 repeat")
            list2_inds.append(curr_ind)
            j += 1
        elif safe_index(list1, i) < safe_index(list2, j):
            # print("1 add")
            curr_ind += 1
            list1_inds.append(curr_ind)
            merged_list.append(list1[i])
            i += 1
        else:
            curr_ind += 1
            list2_inds.append(curr_ind)
            merged_list.append(list2[j])
            # print("2 add")
            j += 1

    return merged_list, list1_inds, list2_inds

def combine_variable_matrix_stacks(var_list_1, var_list_2, var_mat_1, var_mat_2):
    var_list, l1_indices, l2_indices = merge_sorted_lists(var_list_1, var_list_2)
    
    def make_super_matrix_stack(old_mat, submat_indices, new_size):
        new_matrix_stack = np.full((old_mat.shape[0], new_size, new_size), np.nan)
        dim1 = np.ix_(submat_indices, submat_indices)[0]
        dim2 = np.ix_(submat_indices, submat_indices)[1]
        new_matrix_stack[:, dim1, dim2] = old_mat

        return new_matrix_stack

    new_var_mat_1 = make_super_matrix_stack(var_mat_1, l1_indices, len(var_list))
    new_var_mat_2 = make_super_matrix_stack(var_mat_2, l2_indices, len(var_list))

    new_var_mat = np.concatenate((new_var_mat_1, new_var_mat_2), axis=0)

    return var_list, new_var_mat



def make_edge_time_series_from_graph(start_year, end_year, interval, overlap, node1, node2):
    curr_year = start_year
    edge_weights = []
    years = []
    while curr_year + interval <= end_year:
        path = f"../out/belief networks/{curr_year}-{curr_year + interval}, R=0.2, Condition=None"
        G = nx.read_graphml(os.path.join(path, "graph_object.graphml"))
        _, var_list = csv_to_sorted_list(os.path.join(path, "variables_list.csv"))
        
        if node1 in G and node2 in G:
            edge_data = G.get_edge_data(node1, node2)

            if edge_data is None:
                edge_val = 0
            else:
                edge_val = edge_data.get('weight')
        else:
            edge_val = None

        years.append(curr_year)
        edge_weights.append(edge_val)
        curr_year = curr_year + interval - overlap
  

    return edge_weights, years



def make_edge_time_series(start_year, end_year, interval, overlap):
    curr_year = start_year
    years = []
    while curr_year + interval <= end_year:
        path = f"../out/belief networks/{curr_year}-{curr_year + interval}, R=0.2, Condition=None"
        print(path)

        new_data, new_var_list = get_sorted_adj_mat_and_var_list(path)
        new_data = new_data.reshape(1, len(new_var_list), len(new_var_list))

        if start_year != curr_year:
            curr_var_list, curr_data = combine_variable_matrix_stacks(curr_var_list, new_var_list, curr_data, new_data)
        else:
            curr_var_list = new_var_list
            curr_data = new_data

        years.append(curr_year)
        curr_year = curr_year + interval - overlap
  

    return curr_var_list, curr_data, years

def merge_graphs_with_weight_diff(before_graph, after_graph, pct_change=False):
    # Create a new empty graph
    merged_graph = nx.Graph()

    # Add nodes from both graphs to the new graph
    merged_graph.add_nodes_from(before_graph.nodes())
    merged_graph.add_nodes_from(after_graph.nodes())

    # Get the union of edges from both graphs
    all_edges = set(before_graph.edges()).union(set(after_graph.edges()))

    # Add edges with the sum of weights
    for edge in all_edges:
        # Get the weight of the edge in graph1 (0 if the edge doesn't exist)
        
        before_weight = before_graph[edge[0]].get(edge[1], {}).get('weight', 0) if edge[0] in before_graph and edge[1] in before_graph else 0
        
        # Get the weight of the edge in graph2 (0 if the edge doesn't exist)
        after_weight = after_graph[edge[0]].get(edge[1], {}).get('weight', 0) if edge[0] in after_graph and edge[1] in after_graph else 0
        
        # Add the edge with the summed weight to the new graph
        change = (after_weight - before_weight)/((before_weight + after_weight)/2) if pct_change else after_weight - before_weight
        merged_graph.add_edge(edge[0], edge[1], weight=after_weight, type=change)

    return merged_graph

def make_change_graph(adj_var_list, adj_mat, change_var_list, change_mat):
    new_change_mat = make_consistent_matrix(adj_var_list, change_var_list, change_mat)
    num_nodes = adj_mat.shape[0]

    G = nx.Graph()
    G.add_nodes_from(adj_var_list)

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adj_mat[i, j] != 0 or ~np.isnan(new_change_mat[i, j]):
                weight = max(1e-5, adj_mat[i, j])
                change = 0 if np.isnan(new_change_mat[i, j]) else new_change_mat[i, j]

                G.add_edge(adj_var_list[i], adj_var_list[j], weight=weight, type=change)
    
    return G

def calc_sd_change_from_window_mean(matrix_stack, window, start_at=None, reqd_samples=None):
    start_row = window if start_at is None else start_at
    reqd_samples = window if reqd_samples is None else reqd_samples
    change_mat = np.full(matrix_stack.shape, np.nan)
    
    for curr_row in range(start_row, matrix_stack.shape[0]):
        window_start = max(0, curr_row - window)
        relevant_substack = matrix_stack[window_start:curr_row, :, :]
        sufficient_samples = np.sum(~np.isnan(relevant_substack), axis=0) >= reqd_samples

        window_mean = np.where(sufficient_samples, np.nanmean(relevant_substack, axis=0), np.nan)
        window_sdev = np.where(sufficient_samples, np.nanstd(relevant_substack, axis=0), np.nan)
        denominator = np.where(window_sdev != 0, window_sdev, np.nan)
        change_mat[curr_row, :, :] = (matrix_stack[curr_row, :, :] - window_mean) / denominator

    return change_mat



