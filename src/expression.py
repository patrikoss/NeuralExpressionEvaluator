from src import expression_evaluator as evaluator


class ExpressionBox:
    def __init__(self, image, max_row_dist=30, max_col_dist=100):
        self.symbol_boxes = list()
        self.image = image
        self.max_row_dist = max_row_dist
        self.max_col_dist = max_col_dist
        self.top, self.left, self.bottom, self.right = image.shape[0]+1,image.shape[1]+1,-1,-1

    def can_add(self, symbol):
        if not self.symbol_boxes:
            return True
        last_symbol = self.symbol_boxes[-1]
        return abs(symbol.center_row - last_symbol.center_row) <= self.max_row_dist and \
               abs(symbol.center_col - last_symbol.center_col) <= self.max_col_dist

    def add_symbol(self, symbol):
        self.symbol_boxes.append(symbol)
        self.top = min(self.top, symbol.top)
        self.left = min(self.left, symbol.left)
        self.right = max(self.right, symbol.right)
        self.bottom = max(self.bottom, symbol.bottom)


def get_expressions_boxes(symbols_boxes, image):
    """
    symbols is a list of 4-tuples of ints representing coordinates for a single detected symbol:
    (top, left, bottom, right)
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
