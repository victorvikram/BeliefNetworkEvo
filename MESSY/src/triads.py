
def count_triads(adj_mat):
    dim = adj_mat.shape[0]
    negative_count = 0
    positive_count = 0

    for i in range(dim):
        for j in range(i + 1, dim):
            for k in range(j + 1, dim):
                triad_weights_0 = adj_mat[i, j]
                triad_weights_1 = adj_mat[j, k]
                triad_weights_2 = adj_mat[i, k]

                prod_pos = (triad_weights_0 * triad_weights_1 * triad_weights_2 > 0)
                prod_neg = (triad_weights_0 * triad_weights_1 * triad_weights_2 < 0)

                positive_count += prod_pos
                negative_count += prod_neg
    
    return positive_count, negative_count