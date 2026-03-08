#-------------------------------------------------------------
# AUTHOR: Arpit Jaysukhbhai Vora
# FILENAME: assignment_2.py
# SPECIFICATION: Simplified SPIMI-based binary inverted index construction with
#                10 blocks (100 docs each) and a constrained multiway merge.
# FOR: CS 5180- Assignment #2
# TIME SPENT: 4 hours
#-----------------------------------------------------------*/
# importing required libraries
import os
import heapq
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# -----------------------------
# PARAMETERS
# -----------------------------
INPUT_PATH = "corpus/corpus.tsv"
BLOCK_SIZE = 100
NUM_BLOCKS = 10
READ_BUFFER_LINES_PER_FILE = 100
WRITE_BUFFER_LINES = 500


# -----------------------------
# Helper functions
# -----------------------------

def docid_to_int(doc_id):
    """
    Convert docIDs like 'D0001' to integer 1.
    Also supports plain numeric strings like '123'.
    """
    s = str(doc_id).strip()
    if s and (s[0] == "D" or s[0] == "d"):
        s = s[1:]
    return int(s)


def parse_block_line(line):
    """
    Parse a line: term:docID1,docID2,...
    Returns (term, [docIDs...]) where docIDs are ints.
    """
    line = line.rstrip("\n")
    term, postings_str = line.split(":", 1)
    if postings_str == "":
        return term, []
    return term, [int(x) for x in postings_str.split(",") if x]


def format_block_line(term, postings):
    """
    Format (term, postings_list) into 'term:1,2,3'
    """
    return term + ":" + ",".join(map(str, postings))


def merge_sorted_unique(a, b):
    """
    Merge two sorted lists into one sorted list with duplicates removed.
    """
    i, j = 0, 0
    out = []
    last = None

    while i < len(a) and j < len(b):
        if a[i] < b[j]:
            v = a[i]
            i += 1
        elif b[j] < a[i]:
            v = b[j]
            j += 1
        else:
            v = a[i]
            i += 1
            j += 1

        if last != v:
            out.append(v)
            last = v

    while i < len(a):
        v = a[i]
        i += 1
        if last != v:
            out.append(v)
            last = v

    while j < len(b):
        v = b[j]
        j += 1
        if last != v:
            out.append(v)
            last = v

    return out


def fill_read_buffer(file_obj, read_buffer, max_lines):
    """
    Read up to max_lines lines from file_obj into read_buffer (a list of (term, postings)).
    Returns True if any lines were read, else False (EOF).
    """
    read_buffer.clear()
    for _ in range(max_lines):
        line = file_obj.readline()
        if not line:
            break
        read_buffer.append(parse_block_line(line))
    return len(read_buffer) > 0


# ---------------------------------------------------------
# 1) READ FIRST BLOCK OF 100 DOCUMENTS USING PANDAS
# ---------------------------------------------------------
# Use pandas.read_csv with chunksize=100.
# Each chunk corresponds to one memory block.
# Convert docIDs like "D0001" to integers.
# ---------------------------------------------------------
# --> add your Python code here
# ---------------------------------------------------------

def main():
    # Make sure input exists
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Cannot find input file: {INPUT_PATH}")

    # ---------------------------------------------------------
    # 4) REPEAT STEPS 1–3 FOR ALL 10 BLOCKS
    # ---------------------------------------------------------
    # - Continue reading next 100-doc chunks
    # - After processing each block, flush to disk
    # - Do NOT keep previous blocks in memory
    # ---------------------------------------------------------
    # --> add your Python code here
    # ---------------------------------------------------------

    block_files = []


    chunk_iter = pd.read_csv(
        INPUT_PATH,
        sep="\t",
        header=None,
        names=["doc_id", "text"],
        encoding="utf-8",
        chunksize=BLOCK_SIZE,
        dtype={"doc_id": "string", "text": "string"},
        keep_default_na=False
    )


    for block_num, chunk_df in enumerate(chunk_iter, start=1):
        if block_num > NUM_BLOCKS:
            break

        chunk_df["doc_id"] = chunk_df["doc_id"].apply(docid_to_int)

        chunk_df = chunk_df.sort_values("doc_id", ascending=True)

        doc_ids = chunk_df["doc_id"].tolist()
        texts = chunk_df["text"].tolist()

        # ---------------------------------------------------------
        # 2) BUILD PARTIAL INDEX (SPIMI STYLE) FOR CURRENT BLOCK
        # ---------------------------------------------------------
        # - Use CountVectorizer(stop_words='english')
        # - Fit and transform the 100 documents
        # - Reconstruct binary postings lists from the sparse matrix
        # - Store postings in a dictionary: term -> set(docIDs)
        # ---------------------------------------------------------
        # --> add your Python code here
        # ---------------------------------------------------------

        vectorizer = CountVectorizer(stop_words="english", lowercase=True)
        X = vectorizer.fit_transform(texts)  # sparse doc-term matrix
        vocab = vectorizer.get_feature_names_out()

        partial_index = {}  # term -> set(docIDs)


        for row_i, doc_id in enumerate(doc_ids):
            row = X.getrow(row_i)
            term_indices = row.indices  # which terms appear in this doc
            for ti in term_indices:
                term = vocab[ti]
                if term not in partial_index:
                    partial_index[term] = set()
                partial_index[term].add(doc_id)

        # ---------------------------------------------------------
        # 3) FLUSH PARTIAL INDEX TO DISK
        # ---------------------------------------------------------
        # - Sort terms lexicographically
        # - Sort postings lists (ascending docID)
        # - Write to: block_1.txt, block_2.txt, ..., block_10.txt
        # - Format: term:docID1,docID2,docID3
        # ---------------------------------------------------------
        # --> add your Python code here
        # ---------------------------------------------------------

        block_filename = f"block_{block_num}.txt"
        with open(block_filename, "w", encoding="utf-8") as bf:
            for term in sorted(partial_index.keys()):
                postings_sorted = sorted(partial_index[term])
                bf.write(format_block_line(term, postings_sorted) + "\n")

        block_files.append(block_filename)

        del partial_index
        del X
        del vectorizer

    if len(block_files) != NUM_BLOCKS:
        raise ValueError(f"Expected {NUM_BLOCKS} blocks but created {len(block_files)}. "
                         f"Check that corpus has at least {NUM_BLOCKS * BLOCK_SIZE} documents.")

    # ---------------------------------------------------------
    # 5) FINAL MERGE PHASE
    # ---------------------------------------------------------
    # After all block files are created:
    # - Open block_1.txt ... block_10.txt simultaneously
    # ---------------------------------------------------------
    # --> add your Python code here
    # ---------------------------------------------------------

    file_objs = [open(fname, "r", encoding="utf-8") for fname in block_files]

    # ---------------------------------------------------------
    # 6) INITIALIZE READ BUFFERS
    # ---------------------------------------------------------
    # For each block file:
    # - Read up to READ_BUFFER_LINES_PER_FILE lines
    # - Parse each line into (term, postings_list)
    # - Store in a per-file read buffer
    # ---------------------------------------------------------
    # --> add your Python code here
    # ---------------------------------------------------------

    read_buffers = [[] for _ in range(NUM_BLOCKS)]   # list of lists of (term, postings)
    buffer_pos = [0 for _ in range(NUM_BLOCKS)]      # pointer into each read buffer
    has_data = [False for _ in range(NUM_BLOCKS)]    # whether buffer currently has data

    for i in range(NUM_BLOCKS):
        has_data[i] = fill_read_buffer(file_objs[i], read_buffers[i], READ_BUFFER_LINES_PER_FILE)
        buffer_pos[i] = 0

    # ---------------------------------------------------------
    # 7) INITIALIZE MIN-HEAP (OR SORTED STRUCTURE)
    # ---------------------------------------------------------
    # - Push the first term from each read buffer into a min-heap
    # - Heap elements: (term, file_index)
    # ---------------------------------------------------------
    # --> add your Python code here
    # ---------------------------------------------------------

    min_heap = []
    for i in range(NUM_BLOCKS):
        if has_data[i] and buffer_pos[i] < len(read_buffers[i]):
            term_i, _ = read_buffers[i][buffer_pos[i]]
            heapq.heappush(min_heap, (term_i, i))

    # ---------------------------------------------------------
    # 8) MERGE LOOP
    # ---------------------------------------------------------
    # While min-heap is not empty:
    # 1. Pop the min-heap root (smallest term)
    # 2. Keep popping the min-heap root while the current term equals the previous term
    # 3. Collect all read buffers whose current term matches
    # 4. Merge postings lists associated with this term (sorted + deduplicated)
    # 5. Advance corresponding read buffer pointers
    # 6. If a read buffer is exhausted, read next 100 lines from the corresponding block (if available)
    # 7. For each read buffer whose pointer advanced, push its new pointed term into the heap (if available).
    # ---------------------------------------------------------
    # --> add your Python code here
    # ---------------------------------------------------------

    # ---------------------------------------------------------
    # 9) WRITE BUFFER MANAGEMENT
    # ---------------------------------------------------------
    # - Append merged term-line to write buffer
    # - If write buffer reaches WRITE_BUFFER_LINES: flush (append) to final_index.txt
    # - After merge loop ends: flush remaining write buffer
    # ---------------------------------------------------------
    # --> add your Python code here
    # ---------------------------------------------------------

    write_buffer = []
    final_out = open("final_index.txt", "w", encoding="utf-8")

    while min_heap:
        current_term, file_i = heapq.heappop(min_heap)

        postings_accum = []

        def consume_if_matches(i):
            nonlocal postings_accum

            if not has_data[i]:
                return False

            if buffer_pos[i] >= len(read_buffers[i]):
                has_data[i] = fill_read_buffer(file_objs[i], read_buffers[i], READ_BUFFER_LINES_PER_FILE)
                buffer_pos[i] = 0
                if not has_data[i]:
                    return False

            term_i, postings_i = read_buffers[i][buffer_pos[i]]
            if term_i != current_term:
                return False

            postings_accum.append(postings_i)
            buffer_pos[i] += 1
            return True

        consume_if_matches(file_i)

        same_term_files = []
        while min_heap and min_heap[0][0] == current_term:
            _, j = heapq.heappop(min_heap)
            same_term_files.append(j)

        for j in same_term_files:
            consume_if_matches(j)

        merged = []
        for plist in postings_accum:
            merged = merge_sorted_unique(merged, plist)

        write_buffer.append(format_block_line(current_term, merged))

        if len(write_buffer) >= WRITE_BUFFER_LINES:
            final_out.write("\n".join(write_buffer) + "\n")
            write_buffer.clear()

        involved_files = [file_i] + same_term_files
        for i in involved_files:
            if not has_data[i]:
                continue

            if buffer_pos[i] >= len(read_buffers[i]):
                has_data[i] = fill_read_buffer(file_objs[i], read_buffers[i], READ_BUFFER_LINES_PER_FILE)
                buffer_pos[i] = 0

            if has_data[i] and buffer_pos[i] < len(read_buffers[i]):
                next_term, _ = read_buffers[i][buffer_pos[i]]
                heapq.heappush(min_heap, (next_term, i))

    if write_buffer:
        final_out.write("\n".join(write_buffer) + "\n")
        write_buffer.clear()

    # ---------------------------------------------------------
    # 10) CLEANUP
    # ---------------------------------------------------------
    # - Close all open block files
    # - Ensure final_index.txt is properly written
    # ---------------------------------------------------------
    # --> add your Python code here
    # ---------------------------------------------------------

    final_out.close()
    for f in file_objs:
        f.close()

    print("Done.")
    print("Blocks:", ", ".join(block_files))
    print("Final:", "final_index.txt")


if __name__ == "__main__":
    main()