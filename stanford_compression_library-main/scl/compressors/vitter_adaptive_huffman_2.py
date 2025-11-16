from scl.utils.tree_utils import BinaryNode
from scl.compressors.prefix_free_compressors import PrefixFreeTree

class VitterNode(BinaryNode):
    def __init__(self, symbol=None, weight=0, implicit_num=511):
        super().__init__(id=symbol)
        self.weight = weight
        self.parent = None
        self.implicit_num = implicit_num
    
    def is_nyt(self):
        return (self.weight == 0) and (self.is_leaf_node)
    
    def is_root(self):
        return self.parent is None

class VitterAdaptiveHuffmanTree(PrefixFreeTree):
    def __init__(self, alphabet_size=256, symbol_bits=8):
        self.alphabet_size = alphabet_size
        self.symbol_bits = symbol_bits
        self.root_node = VitterNode(implicit_num=2 * alphabet_size - 1) # the maximum number of nodes of a binary tree with n leaves is 2n-1
        self.nyt = self.root_node
        self.leaves = {}        # symbol -> leaf node
        self.blocks_leaf = {0: [self.root_node]}  # weight -> [list of leaf nodes in ascending implicit number]
        self.blocks_internal = {}  # weight -> [list of internal nodes in ascending implicit number]

    # ----------------------------------------------------------
    # Linked-list and block helpers
    # ----------------------------------------------------------
    def insert_after(self, a, b):
        """Insert node b right after node a in global linked list."""
        b.prev = a
        b.next = a.next
        if a.next:
            a.next.prev = b
        else:
            self.tail = b
        a.next = b

    def remove_from_block(self, node):
        """Remove node from its block."""
        wt = node.weight
        if node.is_leaf_node:
            self.blocks_leaf[wt].remove(node)
            if len(self.blocks_leaf[wt]) == 0:
                del self.blocks_leaf[wt]
        else:
            self.blocks_internal[wt].remove(node)
            if len(self.blocks_internal[wt]) == 0:
                del self.blocks_internal[wt]

    def add_to_block(self, node):
        if node.is_leaf_node:
            if node.weight not in self.blocks_leaf:
                self.blocks_leaf[node.weight] = [node]
            else:
                # add to the list according to implicit number
                lst = self.blocks_leaf[node.weight]
                inserted = False
                # the implicit number increases from head to tail
                for i in range(len(lst)):
                    if node.implicit_num < lst[i].implicit_num:
                        lst.insert(i, node)
                        inserted = True
                        break
                if not inserted:
                    lst.append(node)
        else:
            if node.weight not in self.blocks_internal:
                self.blocks_internal[node.weight] = [node]
            else:
                # add to the list according to implicit number
                lst = self.blocks_internal[node.weight]
                inserted = False
                for i in range(len(lst)):
                    if node.implicit_num < lst[i].implicit_num:
                        lst.insert(i, node)
                        inserted = True
                        break
                if not inserted:
                    lst.append(node)
    
    # ----------------------------------------------------------
    # Core algorithm: slide and increment
    # ----------------------------------------------------------
    def slide_and_increment(self, node):
        # reference: Vitter, J. S. (1987). Design and analysis of dynamic Huffman codes. page 839
        prev_parent = node.parent # previous parent
        self.slide_nodes(node)
        self.remove_from_block(node)
        node.weight += 1
        self.add_to_block(node)

        if node.is_leaf_node:
            return node.parent
        else:
            return prev_parent
    
    def update(self, node, sym):
        # reference: Vitter, J. S. (1987). Design and analysis of dynamic Huffman codes. page 839
        leaf2incrment = None
        if node is None:
            # new symbol
            internal = VitterNode(weight=0)
            new_nyt = VitterNode(weight=0)
            leaf = VitterNode(symbol=sym, weight=0)
            internal.left_child, internal.right_child = new_nyt, leaf
            new_nyt.parent = internal
            leaf.parent = internal

            # replace nyt with internal node
            nyt = self.nyt
            if nyt.parent:
                p = nyt.parent
                if p.left_child is nyt:
                    p.left_child = internal
                else:
                    p.right_child = internal
                internal.parent = p
            else:
                self.root_node = internal
            self.nyt = new_nyt
            self.leaves[sym] = leaf

            # insert into linked list
            new_nyt.implicit_num = nyt.implicit_num - 2
            leaf.implicit_num = nyt.implicit_num - 1
            internal.implicit_num = nyt.implicit_num

            self.remove_from_block(nyt)
            self.add_to_block(new_nyt)
            if not internal.is_root():
                self.add_to_block(internal)
            self.add_to_block(leaf)

            q = internal
            leaf2incrment = leaf
        
        else:
            # existing symbol
            q = node
            # interchange q in the tree with the leader of its block
            if q.weight in self.blocks_internal:
                leader = self.blocks_internal[q.weight][-1]
                if leader is not q:
                    self.swap_nodes(q, leader)

            q_sbling = q.parent.left_child if q.parent.right_child is q else q.parent.right_child
            if q_sbling.is_nyt():
                leaf2incrment = q
                q = q.parent
        while not q.is_root():
            q = self.slide_and_increment(q)
        if leaf2incrment:
            self.slide_and_increment(leaf2incrment)

    def slide_nodes(self, a):
        """Slide node a to position of node b in linked list and tree."""
        block_to_slide = None
        if a.is_leaf_node:
            if a.weight in self.blocks_internal:
                block_to_slide = self.blocks_internal[a.weight]
        else:
            if a.weight + 1 in self.blocks_leaf:
                block_to_slide = self.blocks_leaf[a.weight + 1]
        if block_to_slide is not None:
            # only keep nodes with higher implicit number than a
            block_to_slide = [node for node in block_to_slide if node.implicit_num > a.implicit_num]
            for node in block_to_slide:
                self.swap_nodes(a, node)
        

    def swap_nodes(self, a, b):
        """Swap two nodes in tree and linked list (not parent-child or ancestor-descendant)."""
        pa, pb = a.parent, b.parent
        if pa is None or pb is None:
            return  # don't swap root
        
        # Check if nodes are in ancestor-descendant relationship
        # Check if a is an ancestor of b
        temp = b.parent
        while temp:
            if temp is a:
                return  # a is ancestor of b, don't swap
            temp = temp.parent
        # Check if b is an ancestor of a
        temp = a.parent
        while temp:
            if temp is b:
                return  # b is ancestor of a, don't swap
            temp = temp.parent

        # swap in tree
        if pa is not pb:
            if pa.left_child is a:
                pa.left_child = b
            else:
                pa.right_child = b
            if pb.left_child is b:
                pb.left_child = a
            else:
                pb.right_child = a
            a.parent, b.parent = pb, pa
        else:
            # same parent
            if pa.left_child is a:
                pa.left_child, pa.right_child = b, a
            else:
                pa.left_child, pa.right_child = a, b

        # swap implicit numbers
        a.implicit_num, b.implicit_num = b.implicit_num, a.implicit_num

    # ----------------------------------------------------------
    # Encode / Decode
    # ----------------------------------------------------------
    def get_code(self, node):
        bits = []
        while node.parent:
            bits.append('0' if node.parent.left_child is node else '1')
            node = node.parent
        return ''.join(reversed(bits))

    def byte_to_bits(self, b):
        return format(b, f'0{self.symbol_bits}b')

    def bits_to_byte(self, bits):
        return int(bits, 2)

    # ----------------------------------------------------------
    # Snapshot for visualization
    # ----------------------------------------------------------
    def export_tree_snapshot(self):
        if self.root_node is None:
            raise ValueError("Tree is empty")

        nodes = []
        internal_counter = 1

        def _dfs(node, depth):
            nonlocal internal_counter

            if node.is_leaf_node:
                if node.is_nyt():
                    nid = "NYT"
                else:
                    nid = str(node.id)

                w = float(node.weight)

                nodes.append(
                    {
                        "id": nid,
                        "weight": w,
                        "depth": depth,
                        "left": None,
                        "right": None,
                        "is_leaf": True,
                        "implicit_num": getattr(node, "implicit_num", None),
                    }
                )
                return nid, w

            left_id, left_w = _dfs(node.left_child, depth + 1)
            right_id, right_w = _dfs(node.right_child, depth + 1)

            if depth == 0:
                nid = "*"
            else:
                nid = f"N{internal_counter}"
                internal_counter += 1

            total_w = left_w + right_w

            nodes.append(
                {
                    "id": nid,
                    "weight": total_w,
                    "depth": depth,
                    "left": left_id,
                    "right": right_id,
                    "is_leaf": False,
                    "implicit_num": getattr(node, "implicit_num", None),
                }
            )
            return nid, total_w

        root_id, _ = _dfs(self.root_node, 0)
        ordered = list(reversed(nodes))

        return {"root": root_id, "tree": ordered}

class VitterAdaptiveHuffmanEncoder(VitterAdaptiveHuffmanTree):

    def __init__(self, alphabet_size=256, symbol_bits=8):
        super().__init__(alphabet_size=alphabet_size, symbol_bits=symbol_bits)

    def encode(self, data: bytes) -> str:
        bits = []
        for b in data:
            if b in self.leaves:
                bits.append(self.get_code(self.leaves[b]))
                self.update(self.leaves[b], b)
            else:
                bits.append(self.get_code(self.nyt))
                bits.append(self.byte_to_bits(b))
                self.update(None, b)
            # self.print_tree()
        return ''.join(bits)
    
    def encode_with_trace(self, data: bytes):
        bit_chunks = []
        steps = []
        step_idx = 0

        for b in data:
            if b in self.leaves:
                code = self.get_code(self.leaves[b])
                bit_chunks.append(code)
                self.update(self.leaves[b], b)
                action = "existing"
            else:
                code_nyt = self.get_code(self.nyt)
                sym_bits = self.byte_to_bits(b)
                bit_chunks.append(code_nyt + sym_bits)
                self.update(None, b)
                action = "new"

            step_idx += 1
            snapshot = self.export_tree_snapshot()
            for n in snapshot["tree"]:
                n["build_index"] = step_idx

            steps.append(
                {
                    "step": step_idx,
                    "symbol": int(b),
                    "action": action,
                    "tree": snapshot["tree"],
                }
            )

        return {
            "steps": steps,
            "bitstream": ''.join(bit_chunks),
            "max_step": step_idx,
        }
    
class VitterAdaptiveHuffmanDecoder(VitterAdaptiveHuffmanTree):

    def __init__(self, alphabet_size=256, symbol_bits=8):
        super().__init__(alphabet_size=alphabet_size, symbol_bits=symbol_bits)

    def decode(self, bits: str) -> bytes:
        idx = 0
        out = []
        while idx < len(bits):
            node = self.root_node
            depth = 0

            # Traverse down the tree until we reach a leaf
            while not node.is_leaf_node:
                bit = bits[idx]
                idx += 1
                depth += 1
                if bit == '0':
                    node = node.left_child
                else:
                    node = node.right_child
                if node is None:
                    raise ValueError("Invalid bit sequence: reached None node")

            if node.is_nyt():
                # meaning new symbol
                if idx + self.symbol_bits > len(bits):
                    raise ValueError("Not enough bits to read symbol")
                sym_bits = bits[idx:idx + self.symbol_bits]
                idx += self.symbol_bits
                sym = self.bits_to_byte(sym_bits)
                self.update(None, sym)
                out.append(sym)
            else:
                # existing symbol
                sym = node.id
                out.append(sym)
                self.update(node, sym)
        return bytes(out)
