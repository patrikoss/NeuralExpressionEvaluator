def is_operand(symbol):
    return symbol.isdigit()


def prededence_order(symbol):
    prededence_order = {
        '+': 1,
        '-': 1,
        '*': 2,
        '/': 2,
        '^': 3
    }

    return prededence_order[symbol] if symbol in prededence_order else -1


def infix_to_postfix(symbols):
    if len(symbols) > 0 and symbols[0] == '-' or symbols[0] == '+':
        symbols = ['0'] + symbols

    stack = []
    postfix = []
    for symbol in symbols:
        if is_operand(symbol):
            postfix.append(symbol)
        elif symbol == '(':
            stack.append(symbol)
        elif symbol == ')':
            while len(stack) > 0 and stack[-1] != '(':
                postfix.append(stack.pop())
            if len(stack) == 0:
                msg = 'Invalid expression. Cannot convert to postfix.'
                raise Exception(msg)
            # pops '(' character
            stack.pop()
        else:  # operator
            while len(stack) > 0 and prededence_order(symbol) <= prededence_order(stack[-1]):
                postfix.append(stack.pop())

            stack.append(symbol)

    while len(stack) > 0:
        postfix.append(stack.pop())

    return postfix


def evaluate_single_expression(a, b, operation):
    if operation == '+':
        return a + b
    elif operation == '-':
        return a - b
    elif operation == '*':
        return a * b
    elif operation == '/':
        return a / b
    elif operation == '^':
        return a ** b

    msg = 'Invalid expression.'
    raise Exception


def evaluate_postfix(postfix_exp):
    stack = []
    for symbol in postfix_exp:
        if is_operand(symbol):
            stack.append(float(symbol))
        else:
            b = stack.pop()
            a = stack.pop()
            stack.append(evaluate_single_expression(a, b, symbol))

    return stack[-1]


def evaluate(expression):
    return evaluate_postfix(infix_to_postfix(expression))