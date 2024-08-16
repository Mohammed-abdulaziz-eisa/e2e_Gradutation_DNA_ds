def fasta_reader(sequence_file):
    with open(sequence_file, 'r') as file:
        sequence_name = file.readline().strip()[1:]
        sequence = ''.join(line.strip() for line in file)
    return sequence_name, sequence

def miller_myers_similarity(seq1, seq2, match_score=2, mismatch_penalty=-1, gap_penalty=-1):
    # Initialize variables
    len1 = len(seq1)
    len2 = len(seq2)
    max_score = 0
    max_i = 0
    max_j = 0
    
    # Create score matrix
    score_matrix = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    
    # Fill score matrix
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if seq1[i - 1] == seq2[j - 1]:
                score = match_score
            else:
                score = mismatch_penalty
            
            # Extend the match
            score_matrix[i][j] = max(0, 
                                      score_matrix[i - 1][j - 1] + score,
                                      score_matrix[i - 1][j] + gap_penalty,
                                      score_matrix[i][j - 1] + gap_penalty)
            
            # Update maximum score and indices
            if score_matrix[i][j] > max_score:
                max_score = score_matrix[i][j]
                max_i = i
                max_j = j
    
    # Traceback to get the aligned sequences
    aligned_seq1 = ''
    aligned_seq2 = ''
    i = max_i
    j = max_j
    while i > 0 and j > 0 and score_matrix[i][j] > 0:
        if seq1[i - 1] == seq2[j - 1]:
            aligned_seq1 = seq1[i - 1] + aligned_seq1
            aligned_seq2 = seq2[j - 1] + aligned_seq2
            i -= 1
            j -= 1
        elif score_matrix[i][j] == score_matrix[i - 1][j] + gap_penalty:
            aligned_seq1 = seq1[i - 1] + aligned_seq1
            aligned_seq2 = '-' + aligned_seq2
            i -= 1
        else:
            aligned_seq1 = '-' + aligned_seq1
            aligned_seq2 = seq2[j - 1] + aligned_seq2
            j -= 1
    
    return max_score, aligned_seq1, aligned_seq2

# Example usage:
# seq1 = "ATCGTACG"
# seq2 = "ACGTACGC"

# score, aligned_seq1, aligned_seq2 = miller_myers_similarity(seq1, seq2)
# print("Alignment Score:", score)
# print("Aligned Sequence 1:", aligned_seq1)
# print("Aligned Sequence 2:", aligned_seq2)


parent_seq_file = "../data/external/child_1_78216.txt"
child_seq_file = "../data/external/child_1_78216_simailar.txt"

parent_seq_name, parent_seq = fasta_reader(parent_seq_file)
child_seq_name, child_seq = fasta_reader(child_seq_file)

print("Parent Sequence Length:", len(parent_seq_name))
print("Child Sequence Length:", len(child_seq_name))

score, aligned_seq1, aligned_seq2 = miller_myers_similarity(parent_seq_file, child_seq_name)
print("Alignment Score:", score)
print("Aligned Sequence 1:", aligned_seq1)
print("Aligned Sequence 2:", aligned_seq2)


# if __name__ == "__main__":
#     parent_seq_file = "../data/external/child_1_78216.txt"
#     child_seq_file = "../data/external/child_1_78216_simailar.txt"

#     parent_seq_name, parent_seq = fasta_reader(parent_seq_file)
#     child_seq_name, child_seq = fasta_reader(child_seq_file)

#     print("Parent Sequence Length:", len(parent_seq_name))
#     print("Child Sequence Length:", len(child_seq_name))

#     score, aligned_seq1, aligned_seq2 = miller_myers_similarity(parent_seq_file, child_seq_name)
#     print("Alignment Score:", score)
#     print("Aligned Sequence 1:", aligned_seq1)
#     print("Aligned Sequence 2:", aligned_seq2)
    # max_length = max(len(parent_seq_name), len(child_seq_name))
    # similarity_percentage = (alignment_score / max_length) * 100
    # print("Parent Sequence Name:", parent_seq_name)
    # print("Child Sequence Name:", child_seq_name)
    # print("Alignment Score: ", alignment_score)
    # print("Similarity_score" , similarity_percentage)