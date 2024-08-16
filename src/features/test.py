from enum import IntEnum
import numpy as np

class Score(IntEnum):
    MATCH = 1
    MISMATCH = -1
    GAP = -1

class Trace(IntEnum):
    STOP = 0
    LEFT = 1
    UP = 2
    DIAGONAL = 3

def fasta_reader(sequence_file):
    with open(sequence_file, 'r') as file:
        sequence_name = file.readline().strip()[1:]
        sequence = ''.join(line.strip() for line in file)
    return sequence_name, sequence

def smith_waterman(seq1, seq2, match_ratio=2, mismatch_ratio=-1, gap_ratio=-.50):
    row = len(seq1) + 1
    col = len(seq2) + 1
    matrix = np.zeros(shape=(row, col), dtype="uint8")
    tracing_matrix = np.zeros(shape=(row, col), dtype="uint8")
    max_score = -1
    max_index = (-1, -1)

    for i in range(1, row):
        for j in range(1, col):
            match_value = match_ratio if seq1[i - 1] == seq2[j - 1] else mismatch_ratio
            diagonal_score = matrix[i - 1, j - 1] + match_value
            vertical_score = matrix[i - 1, j] + gap_ratio
            horizontal_score = matrix[i, j - 1] + gap_ratio
            matrix[i, j] = max(0, diagonal_score, vertical_score, horizontal_score)

            if matrix[i, j] == 0:
                tracing_matrix[i, j] = Trace.STOP
            elif matrix[i, j] == horizontal_score:
                tracing_matrix[i, j] = Trace.LEFT
            elif matrix[i, j] == vertical_score:
                tracing_matrix[i, j] = Trace.UP
            elif matrix[i, j] == diagonal_score:
                tracing_matrix[i, j] = Trace.DIAGONAL

            if matrix[i, j] >= max_score:
                max_index = (i, j)
                max_score = matrix[i, j]

    aligned_seq1 = ""
    aligned_seq2 = ""
    current_aligned_seq1 = ""
    current_aligned_seq2 = ""
    (max_i, max_j) = max_index

    while tracing_matrix[max_i, max_j] != Trace.STOP:
        if tracing_matrix[max_i, max_j] == Trace.DIAGONAL:
            current_aligned_seq1 = seq1[max_i - 1]
            current_aligned_seq2 = seq2[max_j - 1]
            max_i -= 1
            max_j -= 1
        elif tracing_matrix[max_i, max_j] == Trace.UP:
            current_aligned_seq1 = seq1[max_i - 1]
            current_aligned_seq2 = '-'
            max_i -= 1
        elif tracing_matrix[max_i, max_j] == Trace.LEFT:
            current_aligned_seq1 = '-'
            current_aligned_seq2 = seq2[max_j - 1]
            max_j -= 1

        aligned_seq1 += current_aligned_seq1
        aligned_seq2 += current_aligned_seq2

    aligned_seq1 = aligned_seq1[::-1]
    aligned_seq2 = aligned_seq2[::-1]

    return aligned_seq1, aligned_seq2, max_score

def calculate_alignment_score(seq1, seq2, match_ratio=1, mismatch_ratio=-1, gap_ratio=-1):
    _, _, alignment_score = smith_waterman(seq1, seq2, match_ratio, mismatch_ratio, gap_ratio)
    return alignment_score

if __name__ == "__main__":
    parent_seq_file = "../data/external/child_1_78216.txt"
    child_seq_file = "../data/external/child_1_78216_simailar.txt"

    parent_seq_name, parent_seq = fasta_reader(parent_seq_file)
    child_seq_name, child_seq = fasta_reader(child_seq_file)

    print("Parent Sequence Length:", len(parent_seq_name))
    print("Child Sequence Length:", len(child_seq_name))

    alignment_score = calculate_alignment_score(parent_seq_name, child_seq_name, match_ratio=1, mismatch_ratio=-1, gap_ratio=-1)
    max_length = max(len(parent_seq_name), len(child_seq_name))
    similarity_percentage = (alignment_score / max_length) * 100
    print("Parent Sequence Name:", parent_seq_name)
    print("Child Sequence Name:", child_seq_name)
    print("Alignment Score: ", alignment_score)
    print("Similarity_score" , similarity_percentage)