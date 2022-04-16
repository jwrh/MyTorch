import numpy as np


def clean_path(path):
	""" utility function that performs basic text cleaning on path """

	# No need to modify
	path = str(path).replace("'","")
	path = path.replace(",","")
	path = path.replace(" ","")
	path = path.replace("[","")
	path = path.replace("]","")

	return path


class GreedySearchDecoder(object):

    def __init__(self, symbol_set):
        """
        
        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        """

        self.symbol_set = symbol_set


    def decode(self, y_probs):
        """

        Perform greedy search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
            batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size

        Returns
        -------

        decoded_path [str]:
            compressed symbol sequence i.e. without blanks or repeated symbols

        path_prob [float]:
            forward probability of the greedy path

        """

        decoded_path = []
        blank = 0
        path_prob = 1

        num_symbols, seq_len, batch_size = y_probs.shape
        Paths = []
        Probs = []
        TotalPaths = []
        for b in range(batch_size):
                prob = path_prob
                symbolPath = ["-"] * seq_len
                for t in range(seq_len):
                    currMax = 0
                    curr = "-"
                    for i in range(num_symbols):
                        if y_probs[i][t][b] > currMax:
                            currMax = y_probs[i][t][b]
                            # Take care of blank symbol
                            if i != 0 :
                                curr = self.symbol_set[i-1]
                            else:
                                curr = "-"
                    symbolPath[t] = curr
                    prob = prob * currMax
                Paths.append(symbolPath)
                Probs.append(prob)
        for b in range(batch_size):
            compressedPath = ""
            prev = None
            for t in range(seq_len):
                if prev == None:
                    prev  = None
                if (prev != None and Paths[b][t] == prev):
                    continue
                if Paths[b][t] == "-":
                    prev = None
                    continue
                compressedPath += Paths[b][t]
                prev = Paths[b][t]
            TotalPaths.append(compressedPath)

        decoded_path = TotalPaths[0]
        path_prob = Probs[0]
        # TODO:
        # 1. Iterate over sequence length - len(y_probs[0])
        # 2. Iterate over symbol probabilities
        # 3. update path probability, by multiplying with the current max probability
        # 4. Select most probable symbol and append to decoded_path

        decoded_path = clean_path(decoded_path)

        return decoded_path, path_prob
        

def Init(SymbolSets, y):
    InitialBlankPathScore, InitialPathScore = {}, {}
    # First push the blank into a path-ending-with-blank stack. No symbol has been invoked yet
    path = ""
    InitialBlankPathScore[path] = y[0] # Score of blank at t=1
    InitialPathsWithFinalBlank = set()
    InitialPathsWithFinalBlank.add(path)

    # Push rest of the symbols into a path-ending-with-symbol set, without the blank
    InitialPathsWithFinalSymbol = set()
    for i in range(len(SymbolSets)):
        path = SymbolSets[i]
        InitialPathScore[path] = y[i + 1]
        InitialPathsWithFinalSymbol.add(path)  # set addition
    return InitialPathsWithFinalBlank, InitialPathsWithFinalSymbol, InitialBlankPathScore, InitialPathScore

def ExtendWithBlank(PathsWithTerminalBlank, PathsWithTerminalSymbol, y, BlankPathScore, PathScore):
    UpdatedPathsWithTerminalBlank = set()
    UpdatedBlankPathScore = {}

    # First work on paths with terminal blanks, horizontal transitions
    for path in PathsWithTerminalBlank:
        # Repeating a blank does not change the symbol sequence
        UpdatedPathsWithTerminalBlank.add(path)
        UpdatedBlankPathScore[path] = BlankPathScore[path] * y[0]
    # Then extend paths with terminal symbols by blanks
    for path in PathsWithTerminalSymbol:
        # If there is already an equivalent string in UpdatedPathsWithTerminalBlank
        # simply add the score. If not create a new entry
        if path in UpdatedPathsWithTerminalBlank:
            UpdatedBlankPathScore[path] += PathScore[path] * y[0]
        else:
            UpdatedPathsWithTerminalBlank.add(path)
            UpdatedBlankPathScore[path] = PathScore[path] * y[0]
    return UpdatedPathsWithTerminalBlank, UpdatedBlankPathScore


def ExtendWithSymbol(PathsWithTerminalBlank, PathsWithTerminalSymbol, SymbolSet, y, BlankPathScore, PathScore):
    UpdatedPathsWithTerminalSymbol = set()
    UpdatedPathScore = {}

    # First extend the paths terminating in blanks. This will always create a new sequence
    for path in PathsWithTerminalBlank:
        for i in range(len(SymbolSet)): # Symbolset does not include blanks
            newpath = path + SymbolSet[i]
            UpdatedPathsWithTerminalSymbol.add(newpath)
            UpdatedPathScore[newpath] = BlankPathScore[path] * y[i+1]

    # Next work on paths with terminal symbols
    for path in PathsWithTerminalSymbol:
        for i in range(len(SymbolSet)): # Symbolset does not include blanks
            # Extend the path with every symbol other than blank
            newpath = path if (SymbolSet[i] == path[-1]) else path + SymbolSet[i] # horizontal
            if newpath in UpdatedPathsWithTerminalSymbol: # Already in list, merge paths
                UpdatedPathScore[newpath] += PathScore[path] * y[i+1]
            else: # Create new path
                UpdatedPathsWithTerminalSymbol.add(newpath)
                UpdatedPathScore[newpath] = PathScore[path] * y[i+1]
    return UpdatedPathsWithTerminalSymbol, UpdatedPathScore

def Prune(PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore, BeamWidth):
    PrunedBlankPathScore, PrunedPathScore = {}, {}
    PrunedPathsWithTerminalBlank, PrunedPathsWithTerminalSymbol = set(), set()
    scorelist = []
    # First gather all the relevant scores
    for p in PathsWithTerminalBlank:
        scorelist.append(BlankPathScore[p])
    for p in PathsWithTerminalSymbol:
        scorelist.append(PathScore[p])

    # Sort and find cutoff score that retains exactly BeamWidth paths
    scorelist.sort(reverse=True)
    cutoff = scorelist[BeamWidth] if (BeamWidth < len(scorelist)) else scorelist[-1]

    for p in PathsWithTerminalBlank:
        if BlankPathScore[p] > cutoff:
            PrunedPathsWithTerminalBlank.add(p)
            PrunedBlankPathScore[p] = BlankPathScore[p]

    for p in PathsWithTerminalSymbol:
        if PathScore[p] > cutoff:
            PrunedPathsWithTerminalSymbol.add(p)
            PrunedPathScore[p] = PathScore[p]
    return PrunedPathsWithTerminalBlank, PrunedPathsWithTerminalSymbol, PrunedBlankPathScore, PrunedPathScore

def MergeIdenticalPaths(PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore):
    # All paths with terminal symbosl will remain
    MergedPaths = PathsWithTerminalSymbol
    FinalPathScore = PathScore

    # Paths with terminal blanks will contribute scores to existing identical paths from
    # PathsWithTerminalSymbol if present, or be included in the final set, otherwise
    for p in PathsWithTerminalBlank:
        if p in MergedPaths:
            FinalPathScore[p] += BlankPathScore[p]
        else:
            MergedPaths.add(p)
            FinalPathScore[p] = BlankPathScore[p]
    return MergedPaths, FinalPathScore


class BeamSearchDecoder(object):

    def __init__(self, symbol_set, beam_width):
        """

        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        beam_width [int]:
            beam width for selecting top-k hypotheses for expansion

        """

        self.symbol_set = symbol_set
        self.beam_width = beam_width
        

    def decode(self, y_probs):
        """
        
        Perform beam search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
			batch size for part 1 will remain 1, but if you plan to use your
			implementation for part 2 you need to incorporate batch_size

        Returns
        -------
        
        forward_path [str]:
            the symbol sequence with the best path score (forward probability)

        merged_path_scores [dict]:
            all the final merged paths with their scores

        """

        decoded_path = []
        sequences = [[list(), 1.0]]
        ordered = None
            # Follow the pseudocode from lecture to complete beam search :-)
        PathScore = {} # dict of scores for paths ending with symbols
        BlankPathScore = {} # dict of scores for paths ending with blanks
        num_symbols, seq_len, batch_size = y_probs.shape

        # First time instant: initialize paths with each of the symbols, including blank, using score at t=1
        NewPathsWithTerminalBlank, NewPathsWithTerminalSymbol, NewBlankPathScore, NewPathScore = Init(self.symbol_set, y_probs[:, 0, :])

        # Subsequent time steps
        for t in range(1, seq_len):
            PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore = Prune(NewPathsWithTerminalBlank,
                                                                                            NewPathsWithTerminalSymbol,
                                                                                            NewBlankPathScore, NewPathScore,
                                                                                            self.beam_width)

            NewPathsWithTerminalBlank, NewBlankPathScore =  ExtendWithBlank(PathsWithTerminalBlank, PathsWithTerminalSymbol, y_probs[:, t, :], BlankPathScore, PathScore)

            # Next extend paths by a symbol
            NewPathsWithTerminalSymbol, NewPathScore = ExtendWithSymbol(PathsWithTerminalBlank, PathsWithTerminalSymbol, self.symbol_set, y_probs[:, t, :], BlankPathScore, PathScore)

        # Merge identical paths differing only by the final blank
        MergedPaths, FinalPathScore = MergeIdenticalPaths(NewPathsWithTerminalBlank, NewPathsWithTerminalSymbol, NewBlankPathScore, NewPathScore)


        # Pick the best path
        BestPath = max(FinalPathScore, key=FinalPathScore.get) # Find the path with the best score
        return BestPath, FinalPathScore

        


















        best_path, merged_path_scores = None, None








        # TODO:
        # 1. Iterate over sequence length - len(y_probs[0])
        #    - initialize a list to store all candidates
        # 2. Iterate over 'sequences'
        # 3. Iterate over symbol probabilities
        #    - Update all candidates by appropriately compressing sequences
        #    - Handle cases when current sequence is empty vs. when not empty
        # 4. Sort all candidates based on score (descending), and rewrite 'ordered'
        # 5. Update 'sequences' with first self.beam_width candidates from 'ordered'
        # 6. Merge paths in 'ordered', and get merged paths scores
        # 7. Select best path based on merged path scores, and return      

        # return best_path, merged_path_scores
        raise NotImplementedError
