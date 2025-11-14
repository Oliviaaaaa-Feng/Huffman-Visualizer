from scl.utils.tree_utils import BinaryNode
from scl.compressors.prefix_free_compressors import PrefixFreeTree


class VitterNode(BinaryNode):
    def __init__(self, symbol=None, weight=0):
        super().__init__(id=symbol)
        self.weight = weight
        self.parent = None
        self.prev = None  # linked list previous (implicit order)
        self.next = None  # linked list next (implicit order)

    def is_nyt(self):
        return (self.weight == 0) and (self.is_leaf_node)


class VitterAdaptiveHuffmanTree(PrefixFreeTree):
    def __init__(self, alphabet_size=256, symbol_bits=8):
        self.alphabet_size = alphabet_size
        self.symbol_bits = symbol_bits
        self.root_node = VitterNode()
        self.nyt = self.root_node
        self.leaves = {}  # symbol -> leaf node
        self.blocks = {0: (self.root_node, self.root_node)}  # weight -> (head, tail)
        self.head = self.root_node  # linked list head (lowest order)
        self.tail = self.root_node  # linked list tail (highest order)

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
        head, tail = self.blocks[node.weight]
        if head == tail == node:
            del self.blocks[node.weight]
        elif head == node:
            self.blocks[node.weight] = (node.next, tail)
        elif tail == node:
            self.blocks[node.weight] = (head, node.prev)
        # else it's in the middle: no need to change head/tail

    def add_to_block(self, node):
        """Add node at end of block for its new weight."""
        if node.weight not in self.blocks:
            self.blocks[node.weight] = (node, node)
        else:
            _, tail = self.blocks[node.weight]
            self.insert_after(tail, node)
            self.blocks[node.weight] = (self.blocks[node.weight][0], node)

    # ----------------------------------------------------------
    # Core algorithm: slide and increment
    # ----------------------------------------------------------
    def update(self, node):
        depth = 0
        while node:
            depth += 1

            # Check if block exists before accessing
            if node.weight not in self.blocks:
                # This shouldn't happen, but if it does, add node to block first
                self.add_to_block(node)

            leader = self.blocks[node.weight][1]  # tail of block
            # swap with leader if necessary and not parent
            if leader is not node and leader is not node.parent:
                self.swap_nodes(node, leader)

            # move node to next block
            self.remove_from_block(node)
            node.weight += 1
            self.add_to_block(node)

            parent = node.parent
            # Safety check: if parent is the same as node, we have a cycle
            if parent is node:
                raise ValueError(f"Node {node} has itself as parent - cycle detected")
            node = parent

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
        if pa.left_child is a:
            pa.left_child = b
        else:
            pa.right_child = b
        if pb.left_child is b:
            pb.left_child = a
        else:
            pb.right_child = a
        a.parent, b.parent = pb, pa

        # swap in linked list
        a_prev, a_next = a.prev, a.next
        b_prev, b_next = b.prev, b.next
        if a_next is b:  # adjacent
            a.prev, a.next, b.prev, b.next = b, b_next, a_prev, a
        elif b_next is a:  # adjacent reversed
            b.prev, b.next, a.prev, a.next = a, a_next, b_prev, b
        else:
            if a_prev:
                a_prev.next = b
            if a_next:
                a_next.prev = b
            if b_prev:
                b_prev.next = a
            if b_next:
                b_next.prev = a
            a.prev, a.next, b.prev, b.next = b_prev, b_next, a_prev, a_next

        # adjust head/tail if needed
        if self.head is a:
            self.head = b
        elif self.head is b:
            self.head = a
        if self.tail is a:
            self.tail = b
        elif self.tail is b:
            self.tail = a

    # ----------------------------------------------------------
    # Encode / Decode helpers
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

    def add_symbol(self, sym):
        nyt = self.nyt
        internal = VitterNode(weight=0)
        new_nyt = VitterNode(weight=0)
        leaf = VitterNode(symbol=sym, weight=1)

        internal.left_child, internal.right_child = new_nyt, leaf
        new_nyt.parent = internal
        leaf.parent = internal

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
        self.insert_after(nyt, new_nyt)
        self.insert_after(new_nyt, leaf)
        self.insert_after(leaf, internal)

        # manage blocks
        self.remove_from_block(nyt)
        # Add new nodes to their respective blocks
        # Note: new_nyt and internal both have weight 0
        if 0 not in self.blocks:
            self.blocks[0] = (new_nyt, internal)
        else:
            # Add to existing block
            _, tail = self.blocks[0]
            self.insert_after(tail, new_nyt)
            self.insert_after(new_nyt, internal)
            self.blocks[0] = (self.blocks[0][0], internal)

        # Add leaf to weight 1 block
        if 1 not in self.blocks:
            self.blocks[1] = (leaf, leaf)
        else:
            _, tail = self.blocks[1]
            self.insert_after(tail, leaf)
            self.blocks[1] = (self.blocks[1][0], leaf)

        # update weights
        # Only update if internal is not the root (has a parent)
        # if internal.parent is not None:
        #     self.update(internal)
        # else:
        #     # If internal is the root, we still need to update the leaf
        #     self.update(leaf)
        self.update(internal)

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
                nodes.append(
                    {
                        "id": nid,
                        "weight": float(node.weight),
                        "depth": depth,
                        "left": None,
                        "right": None,
                    }
                )
                return nid

            # internal node
            left_id = _dfs(node.left_child, depth + 1)
            right_id = _dfs(node.right_child, depth + 1)

            if depth == 0:
                nid = "*"
            else:
                nid = f"N{internal_counter}"
                internal_counter += 1

            nodes.append(
                {
                    "id": nid,
                    "weight": float(node.weight),
                    "depth": depth,
                    "left": left_id,
                    "right": right_id,
                }
            )
            return nid

        root_id = _dfs(self.root_node, 0)
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
                self.update(self.leaves[b])
            else:
                bits.append(self.get_code(self.nyt))
                bits.append(self.byte_to_bits(b))
                self.add_symbol(b)
        return ''.join(bits)

    def encode_with_trace(self, data: bytes):
        bit_chunks = []
        steps = []
        step_idx = 0

        for b in data:
            if b in self.leaves:
                code = self.get_code(self.leaves[b])
                bit_chunks.append(code)
                self.update(self.leaves[b])
                action = "existing"
            else:
                code_nyt = self.get_code(self.nyt)
                sym_bits = self.byte_to_bits(b)
                bit_chunks.append(code_nyt + sym_bits)
                self.add_symbol(b)
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
                self.add_symbol(sym)
                out.append(sym)
            else:
                sym = node.id
                out.append(sym)
                self.update(node)
        return bytes(out)
