def add(x, y):
    return x + y


def fabonacci(n):
    if n == 1 or n == 2:
        return n
    elif n > 2:
        return fabonacci(n - 1) + fabonacci(n - 2)


def make_adder(n):
    def adder(x):
        return x + n

    return adder


def square(x):
    return x * x


def compose(f, g):
    def h(x):
        return f(g(x))

    return h


lambda: print("你是谁?")

curry = lambda f: lambda x: lambda y: f(x, y)


def print_all(x):
    print(x)

    def sum(y):
        return print_all(x + y)

    return sum


print_all(1)(3)(5)

"""
79
"""


def luhn(n, sum=0, odd=True, c=True):
    length = len(str(n))
    if length % 2 == 0 and c:
        odd = not odd
    if odd:
        f1sum, n = n % 10, n // 10
    else:
        f1sum, n = n % 10 * 2, n // 10
    print(n, f1sum, sum, odd)
    if f1sum > 9:
        f1sum = f1sum % 10 + f1sum // 10
    if n > 0:
        return luhn(n, sum + f1sum, not odd, False)
    else:
        return sum + f1sum


def spilt(n):
    return n % 10, n // 10


def luhn_single(n):

    if n < 10:
        return n
    else:
        all_least, last = spilt(n)
        return luhn_double(all_least) + last


def luhn_double(n):
    all_least, last = spilt(n)
    dlast = last * 2
    if dlast > 9:
        dlast = dlast % 10 + dlast // 10
    else:
        return luhn_single(all_least) + dlast


def cascade(n):
    if n < 9:
        print(n)
    else:
        print(n)
        cascade(n // 10)
        print(n)


def grow(n):
    if n < 10:
        print(n)
    else:
        grow(n // 10)
        print(n)


def decline(n):
    if n > 10:
        print(n)
        decline(n // 10)
    else:
        print(n)


def reverse_cascade(n):
    grow(n)
    decline(n)


def fabonacci(n):
    if n == 1 or n == 2:
        return 1
    else:
        return fabonacci(n - 1) + fabonacci(n - 2)


def count_partitions(n, k):
    if n < 0:
        return 0
    elif k == 0:
        return 0
    elif n == 0:
        return 1
    else:
        return count_partitions(n - k, k) + count_partitions(n, k - 1)


def partitions(n, m):
    if n > 0 and m > 0:
        if n == m:
            yield str(m)
        else:
            for i in partitions(n - m, m):
                yield i + " + " + str(m)
            yield from partitions(n, m - 1)


range(5)
for i in range(5):
    print(i)


def sum_list(s):
    if len(s) == 0:
        return 0
    else:
        return sum_list(s[1:]) + s[0]


def large(s, n):
    if s == []:
        return []
    elif s[0] > n:
        return large(s[1:], n)
    else:
        first = s[0]
        with_s0 = large(s[1:], n - first) + [first]
        without_s0 = large(s[1:], n)
        if sum_list(with_s0) > sum_list(without_s0):
            return with_s0
        else:
            return without_s0


def rational(n, d):
    return [n, d]


def numer(x):
    return x[0]


def denom(x):
    return x[1]


"""截然不同的有理化和取分子分母方式"""


def numer(x):
    return x("n")


def denom(x):
    return x("d")


def rational(n, d):
    def select(x):
        if x == "n":
            return n
        else:
            return d

    return select


def yuefen(x):
    nx = numer(x)
    dx = denom(x)
    while dx != 0:
        nx, dx = dx, nx % dx
    return [numer(x) // nx, denom(x) // nx]


def add_rational(x, y):
    return yuefen([numer(x) * denom(y) + numer(y) * denom(x), denom(x) * denom(y)])


def mul_rational(x, y):
    return yuefen([numer(x) * numer(y), denom(x) * denom(y)])


def div_rational(x, y):
    return yuefen([numer(x) * denom(y), denom(x) * numer(y)])


def istree(tree):
    if type(tree) != list or len(tree) < 1:
        return False
    for branch in branches(tree):
        if not istree(branch):
            return False
    return True


def tree(label, branches=[]):
    for branch in branches:
        assert istree(branch)
    return [label] + branches


def label(tree):
    return tree[0]


def branches(tree):
    return tree[1:]


def fib_tree(n):
    if n == 0 or n == 1:
        return tree(n)
    else:
        left, right = fib_tree(n - 1), fib_tree(n - 2)
        return tree(label(left) + label(right), [left, right])


def count_leaves(tree):

    if branches(tree) == []:
        return 1
    else:
        sum = 0
        for branch in branches(tree):
            sum += count_leaves(branch)
        return sum


t = tree(3, [tree(-1), tree(1, [tree(2, [tree(1)]), tree(3)]), tree(1, [tree(-1)])])


def parent_leaves(tree, sum=0):
    if branches(tree) == []:
        print("节点", label(tree), "父节点和:", sum)
        return sum
    else:
        current = 0
        for branch in branches(tree):
            current += parent_leaves(branch, sum + label(tree))
        return current


def sum_leaves(tree, sum=0, order=0):
    current = 0
    "代表当前路径总和"
    sum += label(tree)
    "一致则加1"
    if order == sum:
        current = 1
    for branch in branches(tree):
        current += sum_leaves(branch, sum, order)
    return current


dict = {"one": 1, "two": 2, "three": 3}
dit = iter(dict)


def palindrome(s):
    t = list(zip(s, reversed(s)))
    for i in t:
        if i[0] != i[1]:
            return False
    return True


def min_abs(s):
    t = [abs(i) for i in s]
    min = t[0]
    for i in t:
        min = i
        for j in t[1:]:
            if j < min:
                min = j
    index = 0
    for i in t:
        if min == i:
            yield index
        index += 1


def max_adjsum(s):
    sum = [s[i] + s[i + 1] for i in range(len(s) - 1)]
    for i in sum:
        max = i
        for j in sum[1:]:
            if j > max:
                max = j
    print("最大相邻和:", max)


def houzhui(s):
    t = []
    for i in s:
        if i % 10 not in t:
            t.append(i % 10)
    sorted_t = sorted(t)
    print(sorted_t)
    dict = {x: [] for x in sorted_t}
    for i in s:
        dict[i % 10].append(i)
    return dict


def repeat_element(s):
    dict = {x: 0 for x in s}
    for i in s:
        dict[i] += 1
    for key, value in dict.items():
        if value <= 1:
            yield str(key) + "只出现了一次"
        else:
            yield str(key) + "出现了" + str(value) + "次"


def delay(arg):
    print("delayed")

    def g():
        return arg

    return g


def horse(mask):
    horse = mask

    def mask(horse):
        return horse

    return horse(mask)


mask = lambda horse: horse(2)
horse(mask)
"""
    mask现在是个函数
    horse函数内部
    horse = function
"""


def remove(s: int, n: int) -> int:
    single_num = []
    while s > 0:
        single_num.append(s % 10)
        s = s // 10
    for i in single_num:
        if i == n:
            single_num.remove(i)
    final = reversed(single_num)
    result = 0
    for i in final:
        result = result * 10 + i
    return result


t = [1, 3, 5, 5]
print(t[:-2])


class Account:
    interest = 0.02

    def __init__(self, account_holder):
        self.balance = 0
        self.account_holder = account_holder

    def deposit(self, number):
        self.balance += number

    def withdraw(self, number):
        if self.balance < number:
            print("The balance is not enough")
            return self.balance
        else:
            self.balance -= number
            print("balance:", self.balance)


class CheckingAccount(Account):
    withdraw_fee = 1
    interest = 0.01

    def withdraw(self, number):
        return Account.withdraw(self, number + self.withdraw_fee)


class Bank:
    account_list = []

    def __init__(self):
        self.account_list = []

    def open_account(self, account_holder, balance, sign=Account):
        if sign == Account:
            account = Account(account_holder)
            account.deposit(balance)
            account.type = Account
        else:
            account = CheckingAccount(account_holder)
            account.deposit(balance)
            account.type = CheckingAccount
        self.account_list.append(account)
        return account

    def pay_interest(self):
        for i in self.account_list:
            i.deposit(i.balance * i.interest)
        print("Interest has accumulated for all accounts")


"""
a:{z = -1, f(x)}
b = B(1){n = 4, z = B(0)
f(1):return B(0)
B(0):{n = 4, z = c(y + 1)}
C(1):{n = 4, f(x)}
}
C(2).n = 4
a.z == C.z ? False
a.z == b.z ? False
b.z.z == C(1)

"""
def min_num(n, m):
    if n > 0:
        n, m = m, n % m
    return m

class Tree:
    label = None
    left = None
    right = None
    def __init__(self, label, left = None, right = None):
        self.label = label
        self.left = left
        self.right = right
    
    def is_leave(self):
        if self.left == None and self.right == None:
            return True
        return False

    def height(self):
        if self.is_leave():
            return 0
        else:
            return 1 + max(self.left.height(), self.right.height()) 

    def print_all(self):
        high = self.height()
        print(' ' * high + str(self.label))
        if self.left != None:
            self.left.print_all()
        if self.right != None:
            self.right.print_all()


class Link:
    def __init__(self, first, rest = None):
        assert rest is None or isinstance(rest, Link)
        self.first = first 
        self.rest = rest
    
    def print_all(self):
        print(f'当前节点：{self.first}')
        if self.rest is not None:
            self.rest.print_all()

def range_link(start, end):
    lt = range(start, end)
    if start >= end:
        return None
    else:
        return Link(lt[0], range_link(lt[0] + 1, end))

def map_link(lk, f):
    if lk is None:
        return None
    else:
        return Link(f(lk.first), map_link(lk.rest, f))

def filter_link(lk, f):
    if lk is None:
        return None
    else:
        if f(lk.first):
            return Link(lk.first, filter_link(lk.rest, f))
        else:
            return filter_link(lk.rest, f)

def odd(x = 0):
    if x % 2 == 1:
        return True
    else :
        return False

def add_link(lk, x):
    if lk is None:
        return Link(x)
    else:
        pre = lk.first
        beh = lk.rest.first
        if pre < x < beh:
            t = Link(x, lk.rest)
            lk.rest = t
        else:
            add_link(lk.rest, x)

def fib(n):
    if n == 0 or n == 1:
        return n
    else:
        return fib(n - 1) + fib(n - 2)

def ceshi(f):
    def counted(n):
        counted.count += 1
        return f(n)
    counted.count = 0
    return counted

def mom(f):
    memory = {}
    def counted(n):
        if n not in memory:
            memory[n] = f(n)
        return memory[n]
    return counted
    