class ExpressionBox:
    def __init__(self, image, max_row_dist=30, max_col_dist=100):
        """
        :param image: image on which the expression is found
        :param max_row_dist: maximum distance in height to
                consider two symbols to belong to the same expression
        :param max_col_dist:maximum distance in width to
                consider two symbols to belong to the same expression
        """
        self.symbol_boxes = list()
        self.image = image
        self.max_row_dist = max_row_dist
        self.max_col_dist = max_col_dist
        self.top, self.left, self.bottom, self.right = image.shape[0]+1,image.shape[1]+1,-1,-1

    def can_add(self, symbol):
        """
        Returns True iff the given symbol can be added to the current expression
        """
        if not self.symbol_boxes:
            return True
        last_symbol = self.symbol_boxes[-1]
        return abs(symbol.center_row - last_symbol.center_row) <= self.max_row_dist and \
               abs(symbol.center_col - last_symbol.center_col) <= self.max_col_dist

    def add_symbol(self, symbol):
        """
        Updates the expression structure to contain the given symbol
        """
        self.symbol_boxes.append(symbol)
        self.top = min(self.top, symbol.top)
        self.left = min(self.left, symbol.left)
        self.right = max(self.right, symbol.right)
        self.bottom = max(self.bottom, symbol.bottom)


def get_expressions_boxes(symbols_boxes, image):
    """
    Finds expressions in the image given found symbols.
    :param image: image on which the expression is found
    :param symbols_boxes: list of 4-tuples of ints representing
            coordinates for a single detected symbol
            (top, left, bottom, right)
    :returns: List of ExpressionBox
    """
    symbols_boxes = sorted(symbols_boxes, key=lambda sb:(sb.center_col, sb.center_row))
    expressions = list()
    for sb in symbols_boxes:
        for exp_nr, exp in enumerate(expressions):
            if exp.can_add(sb):
                exp.add_symbol(sb)
                break
        else:
            new_exp = ExpressionBox(image)
            new_exp.add_symbol(sb)
            expressions.append(new_exp)
    return expressions
