MOD = 0xFFFFFFFF00000001
ROOT = 7
L = 0


def fast_pow(a, k):
    base = 1
    while (k):
        if k & 1:
            base = (base * a) % MOD
        a = (a * a) % MOD
        k >>= 1

    return base % MOD


def NTT(rev, data, paddedN, isInverse):
    for i in range(paddedN):
        if i < rev[i]:
            data[i], data[rev[i]] = data[rev[i]], data[i]

    for k in range(1, L + 1):
        mid = 1 << (k - 1)
        wn = fast_pow(ROOT, ((MOD - 1) >> k))
        if isInverse:
            wn = fast_pow(wn, MOD - 2)
        for j in range(0, paddedN, mid << 1):
            w = 1
            for k in range(mid):
                x = data[j + k]
                y = (w * data[j + k + mid]) % MOD
                data[j + k] = (x + y) % MOD
                data[j + k + mid] = (x - y + MOD) % MOD
                w = (w * wn) % MOD


def polynomialMultiply(coeffA, degreeA, coeffB, degreeB):
    global L
    degreeLimit = degreeA + degreeB
    paddedDegreeSize = 1
    while paddedDegreeSize <= degreeLimit:
        paddedDegreeSize <<= 1
        L += 1

    rev = [0] * paddedDegreeSize
    for i in range(paddedDegreeSize):
        rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << (L - 1))

    tempA = [0] * paddedDegreeSize
    tempB = [0] * paddedDegreeSize
    tempA[:degreeA + 1] = coeffA[:]
    tempB[:degreeB + 1] = coeffB[:]

    NTT(rev, tempA, paddedDegreeSize, False)
    NTT(rev, tempB, paddedDegreeSize, False)

    result = [0] * paddedDegreeSize
    for i in range(paddedDegreeSize):
        result[i] = (tempA[i] * tempB[i]) % MOD
    NTT(rev, result, paddedDegreeSize, True)

    inv = fast_pow(paddedDegreeSize, MOD - 2)
    for i in range(paddedDegreeSize):
        result[i] = (result[i] * inv) % MOD

    return result

def read_input_data(filename):
    with open(filename, 'r') as file:
        # 读取第一行
        line = file.readline().strip().split()
        degreeA, degreeB = map(int, line)

        # 读取第二行
        line = file.readline().strip().split()
        coeffA = list(map(int, line))

        # 读取第三行
        line = file.readline().strip().split()
        coeffB = list(map(int, line))

    return degreeA, degreeB, coeffA, coeffB


def compare_lists(filename, listB):
    with open(filename, 'r') as file:
        line = file.readline().strip().split()
        listA = list(map(int, line))

    if len(listA) != len(listB):
        return False

    for a, b in zip(listA, listB):
        if a != b:
            return False

    return True


if __name__ == "__main__":
    degreeA, degreeB, coeffA, coeffB = read_input_data("input.txt")
    # print("degreeA:", degreeA)
    # print("degreeB:", degreeB)
    # print("coeffA:", coeffA)
    # print("coeffB:", coeffB)

    result = polynomialMultiply(coeffA, degreeA, coeffB, degreeB)

    print(result)
