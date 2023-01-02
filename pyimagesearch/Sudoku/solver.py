import copy as cp


# verifier qu'un chiffre n'existe pas dans une ligne
def notInLine(grille, i, k):
    return all(grille[i][j] != k for j in range(9))


# verifier qu'un chiffre n'existe pas dans une colonne
def notInColumn(grille, j, k):
    for i in range(9):
        if grille[i][j] == k:
            return False
    return True


# verifier qu'un chiffre n'existe pas dans un bloc 3*3
def notInBloc(grille, i, j, k):
    _i = i - (i % 3)
    _j = j - (j % 3)
    for i in range(_i, _i + 3):
        for j in range(_j, _j + 3):
            if grille[i][j] == k:
                return False
    return True


# Validation de la grille
def isValid(grille, position):
    if position == 9 * 9:
        return True

    # on recupere les coordonnees de la case
    i = position // 9
    j = position % 9

    # si la case n'est pas vide, on passe à la suivante(appel recursif)
    if grille[i][j] != 0:
        return isValid(grille, position + 1)

    # backtracking
    for k in range(1, 10):
        if notInLine(grille, i, k) and notInColumn(grille, j, k) and notInBloc(grille, i, j, k):
            # on enregistre k dans la grille
            grille[i][j] = k
            # on fait un appel recursif
            if isValid(grille, position + 1):
                return True

    grille[i][j] = 0
    return False


def solver(grille):
    answer_grille = cp.deepcopy(grille)
    return answer_grille if isValid(answer_grille, 0) else "no"
