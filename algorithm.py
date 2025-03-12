import random
import numpy as np


def fila_duplicada(valores, fila, valor):
    """
    Verifica si hay un duplicado del valor en la fila especificada.

    :param valores: La matriz que representa el estado actual del Sudoku.
    :param fila: Índice de la fila a verificar.
    :param valor: El valor que se está comprobando.
    :return: True si hay un duplicado, False si no lo hay.
    """
    for columna in range(0, NB):
        if valores[fila][columna] == valor:
            return True
    return False


def columna_duplicada(valores, columna, valor):
    """
    Verifica si hay un duplicado del valor en la columna especificada.

    :param valores: La matriz que representa el estado actual del Sudoku.
    :param columna: Índice de la columna a verificar.
    :param valor: El valor que se está comprobando.
    :return: True si hay un duplicado, False si no lo hay.
    """
    for fila in range(0, NB):
        if valores[fila][columna] == valor:
            return True
    return False


def bloque_duplicado(valores, fila, columna, valor):
    """
    Verifica si hay un duplicado del valor en el bloque 3x3 al que pertenece la celda especificada.

    :param valores: La matriz que representa el estado actual del Sudoku.
    :param fila: Índice de la fila de la celda a verificar.
    :param columna: Índice de la columna de la celda a verificar.
    :param valor: El valor que se está comprobando.
    :return: True si hay un duplicado, False si no lo hay.
    """
    i = 3 * (int(fila / 3))
    j = 3 * (int(columna / 3))

    for x in range(3):
        for y in range(3):
            if valores[i + x][j + y] == valor:
                return True
    return False


def numero_fila_duplicadas(valores, fila, valor):
    """
    Determina que valor se encunetra duplicado en la fila especificada.

    :param valores: La matriz que representa el estado actual del Sudoku.
    :param fila: Índice de la fila a verificar.
    :param valor: El valor que se está comprobando.
    :return: True si hay un duplicado, False si no lo hay.
    """
    i = 0
    for columna in range(0, NB):
        if valores[fila][columna] == valor:
            i += 1
            if i > 1:
                return True
    return False


def numero_columna_duplicadas(valores, columna, valor):
    """
    Determina que valor se encunetra duplicado en la ccolumna especificada.

    :param valores: La matriz que representa el estado actual del Sudoku.
    :param columna: Índice de la fila a verificar.
    :param valor: El valor que se está comprobando.
    :return: True si hay un duplicado, False si no lo hay.
    """
    i = 0
    for fila in range(0, NB):
        if valores[fila][columna] == valor:
            i += 1
            if i > 1:
                return True
    return False


def numero_bloque_duplicados(valores, fila, columna, valor):
    """
    Verifica si un valor específico aparece más de una vez en un bloque 3x3 de un Sudoku.

    Esta función se utiliza para validar los bloques en un algoritmo genético para resolver Sudokus.
    Un bloque es una subdivisión de 3x3 dentro del tablero de Sudoku.

    Parámetros:
    - valores: Una matriz 2D que representa el tablero de Sudoku.
    - fila: El índice de la fila (base 0) donde se quiere realizar la verificación.
    - columna: El índice de la columna (base 0) donde se quiere realizar la verificación.
    - valor: El valor específico que se quiere verificar en el bloque.

    Retorna:
    - True si el valor aparece más de una vez en el bloque especificado.
    - False si el valor no aparece más de una vez en el bloque.
    """
    p = 0
    i = 3 * (int(fila / 3))
    j = 3 * (int(columna / 3))

    for x in range(3):
        for y in range(3):
            if valores[i + x][j + y] == valor:
                p += 1
                if p > 1:
                    return True
    return False


rand_num = random.randint(1, 1000)
print("Semilla: %d" % rand_num)
random.seed(rand_num)


def cruza(padre1, padre2, tasa_cruce):
    """
    Realiza el cruce entre dos soluciones de Sudoku (padre1 y padre2) para generar dos nuevas soluciones (hijos).

    Este método utiliza una variante del crossover de mapeo, donde se intercambian filas completas entre los padres.
    La probabilidad de que ocurra el cruce está determinada por la 'tasa_cruce'.

    Parámetros:
    - padre1: Una instancia de la clase Poblacion que representa la primera solución de Sudoku.
    - padre2: Una instancia de la clase Poblacion que representa la segunda solución de Sudoku.
    - tasa_cruce: Un número flotante entre 0 y 1 que indica la probabilidad de que ocurra el cruce.

    Retorna:
    - hijo1, hijo2: Dos nuevas instancias de la clase Poblacion, cada una representando una nueva solución de Sudoku.
    """
    hijo1 = Poblacion()
    hijo2 = Poblacion()

    # Hacer una copia de los genes del padre.
    hijo1.valores = np.copy(padre1.valores)
    hijo2.valores = np.copy(padre2.valores)

    r = random.uniform(0, 1.1)
    while r > 1:
        r = random.uniform(0, 1.1)

    # Realizar el cruce.
    if r < tasa_cruce:
        # Seleccionar un punto de cruce.
        punto_cruce1 = random.randint(0, 8)
        punto_cruce2 = random.randint(1, 9)
        while punto_cruce1 == punto_cruce2:
            punto_cruce1 = random.randint(0, 8)
            punto_cruce2 = random.randint(1, 9)

        if punto_cruce1 > punto_cruce2:
            temp = punto_cruce1
            punto_cruce1 = punto_cruce2
            punto_cruce2 = temp

        for i in range(punto_cruce1, punto_cruce2):
            hijo1.valores[i], hijo2.valores[i] = cruza_de_filas(hijo1.valores[i], hijo2.valores[i])

    return hijo1, hijo2


def valores_restantes(fila_padre, restantes):
    """
    Encuentra el índice del primer valor en 'fila_padre' que esté presente en 'restantes'.

    Parámetros:
    - fila_padre: Lista de valores correspondiente a una fila de Sudoku.
    - restantes: Conjunto de valores que aún no han sido utilizados en la fila.

    Retorna:
    - El índice del primer valor en 'fila_padre' que está en 'restantes', o -1 si no hay coincidencias.
    """
    for i in range(0, NB):
        if fila_padre[i] in restantes:
            return i
    return -1


def encontrar_valor(fila_padre, valor):
    """
    Encuentra el índice del valor dado en 'fila_padre'.

    Parámetros:
    - fila_padre: Lista de valores correspondiente a una fila de Sudoku.
    - valor: Valor a buscar en la fila.

    Retorna:
    - El índice del valor en 'fila_padre', o -1 si el valor no está presente.
    """
    for i in range(0, NB):
        if fila_padre[i] == valor:
            return i
    return -1


def actualizar_aptitud(poblacion):
    """
    Actualiza la función de aptitud para cada individuo en la población.

    Parámetros:
    - poblacion: Lista de individuos en la población actual.

    Retorna:
    - La población con las aptitudes actualizadas.
    """
    for individuo in poblacion:
        individuo.funcion_aptitud()
    return poblacion


def torneo(poblacion, tasa_seleccion):
    """
    Realiza una selección tipo torneo entre dos individuos aleatorios de la población.

    Parámetros:
    - poblacion: Lista de individuos en la población actual.
    - tasa_seleccion: Probabilidad de seleccionar el mejor individuo entre los dos.

    Retorna:
    - El individuo ganador del torneo.
    """
    candidato1 = poblacion[random.randint(0, len(poblacion) - 1)]
    candidato2 = poblacion[random.randint(0, len(poblacion) - 1)]
    aptitud1 = candidato1.aptitud
    aptitud2 = candidato2.aptitud

    # Encontrar el mejor y el peor.
    if aptitud1 > aptitud2:
        mejor = candidato1
        peor = candidato2
    else:
        mejor = candidato2
        peor = candidato1

    r = random.uniform(0, 1.1)
    while r > 1:
        r = random.uniform(0, 1.1)
    if r < tasa_seleccion:
        return mejor
    else:
        return peor


def contar_columnas_incorrectas(valores):
    """
    Cuenta el número de columnas incorrectas en un tablero de Sudoku.

    Parámetros:
    - valores: Una matriz 2D que representa el tablero de Sudoku.

    Retorna:
    - El número de columnas incorrectas en el tablero.
    """
    columnas_incorrectas = 0
    for columna in range(9):
        for valor in range(1,10):
            if numero_columna_duplicadas(valores, columna, valor):
                columnas_incorrectas += 1
                break
    return columnas_incorrectas


def contar_filas_incorrectas(valores):
    """
    Cuenta el número de filas incorrectas en un tablero de Sudoku.

    Parámetros:
    - valores: Una matriz 2D que representa el tablero de Sudoku.

    Retorna:
    - El número de filas incorrectas en el tablero.
    """
    filas_incorrectas = 0
    for row in range(9):
        for valor in range(1,10):
            if numero_fila_duplicadas(valores, row, valor):
                filas_incorrectas += 1
                break
    return filas_incorrectas


def contar_bloques_incorrectos(valores):
    """
    Cuenta el número de bloques 3x3 incorrectos en un tablero de Sudoku.

    Parámetros:
    - valores: Una matriz 2D que representa el tablero de Sudoku.

    Retorna:
    - El número de bloques 3x3 incorrectos en el tablero.
    """
    bloques_incorrectos = 0
    for block in range(3):
        for columna in range(3):
            for valor in range(1,10):
                if numero_bloque_duplicados(valores, block * 3, columna * 3, valor):
                    bloques_incorrectos += 1
                    break
    return bloques_incorrectos


class Poblacion(object):
    """
    Representa una solución individual dentro de una población en un algoritmo genético para resolver Sudokus.

    Atributos:
    - valores: Matriz 2D de enteros que representa el tablero de Sudoku.
    - aptitud: Un valor flotante que representa la aptitud de esta solución.
    """
    def __init__(self):
        """
        Inicializa una nueva instancia de la clase Poblacion con un tablero de Sudoku vacío y aptitud inicial cero.
        """
        self.valores = np.zeros((NB, NB), dtype=int)
        self.aptitud = 0.0
        return

    def funcion_aptitud(self):
        """
        Calcula y actualiza la aptitud de esta solución de Sudoku.

        La aptitud se basa en el número de columnas, filas y bloques 3x3 incorrectos, así como la cantidad de ceros (espacios sin llenar) en el tablero.
        """

        columnas_incorrectas = contar_columnas_incorrectas(self.valores)
        filas_incorrectas = contar_filas_incorrectas(self.valores)
        bloques_incorrectos = contar_bloques_incorrectos(self.valores)
        cantidad_ceros = np.count_nonzero(self.valores == 0)


        # Calcular la aptitud basada en el número de columnas, filas y bloques incorrectos
        self.aptitud = 1 - (columnas_incorrectas + filas_incorrectas + bloques_incorrectos) / 27 - cantidad_ceros / 81

        return

    def mutar(self, tasa_mutacion, valores):
        """
        Realiza una mutación en el tablero de Sudoku de esta solución con una probabilidad dada por 'tasa_mutacion'.

        Parámetros:
        - tasa_mutacion: Probabilidad de que ocurra una mutación.
        - valores: Matriz 2D que representa el tablero de Sudoku con el que se compara durante la mutación.

        Retorna:
        - True si la mutación fue exitosa, False de lo contrario.
        """
        r = random.uniform(0, 1.1)
        while r > 1:
            r = random.uniform(0, 1.1)

        exito = False
        if r < tasa_mutacion:
            for _ in range(2):
                # Mutar.
                while not exito:
                    fila1 = random.randint(0, 8)
                    fila2 = random.randint(0, 8)

                    desde_columna = random.randint(0, 8)
                    a_columna = random.randint(0, 8)
                    while desde_columna == a_columna:
                        desde_columna = random.randint(0, 8)
                        a_columna = random.randint(0, 8)

                    if (not columna_duplicada(valores, a_columna, self.valores[fila1][desde_columna]) and
                            not columna_duplicada(valores, desde_columna, self.valores[fila2][a_columna]) and
                            not bloque_duplicado(valores, fila2, a_columna, self.valores[fila1][desde_columna]) and
                            not bloque_duplicado(valores, fila1, desde_columna, self.valores[fila2][a_columna])):
                        temp = self.valores[fila2][a_columna]
                        self.valores[fila2][a_columna] = self.valores[fila1][desde_columna]
                        self.valores[fila1][desde_columna] = temp
                        exito = True

        return exito


def crear_poblacion(Nc, valores):
    """
    Crea una población inicial de soluciones para un Sudoku.

    Parámetros:
    - Nc: Número de individuos en la población.
    - valores: Matriz 2D con los valores iniciales del Sudoku (0 para celdas vacías).

    Retorna:
    - Una lista de individuos (instancias de Poblacion) que forman la población inicial.
    """
    poblacion = []

    # Determinar los valores legales que cada cuadro puede tomar.
    aux = Poblacion()
    aux.valores = [[[] for j in range(0, NB)] for i in range(0, NB)]
    for fila in range(0, NB):
        for columna in range(0, NB):
            for valor in range(1, 10):
                if ((valores[fila][columna] == 0) and not (columna_duplicada(valores, columna, valor) or
                                                           bloque_duplicado(valores, fila, columna, valor) or
                                                           fila_duplicada(valores, fila, valor))):
                    # El valor está disponible.
                    aux.valores[fila][columna].append(valor)
                elif (valores[fila][columna] != 0):
                    aux.valores[fila][columna].append(valores[fila][columna])
                    break

    # Inicializar una nueva población.
    for p in range(0, Nc):
        c = Poblacion()
        for i in range(0, NB):
            fila = np.zeros(NB)

            for j in range(0, NB):

                # Si el valor ya está dado, no lo cambies.
                if valores[i][j] != 0:
                    fila[j] = valores[i][j]
                # Rellenar los espacios en blanco
                elif valores[i][j] == 0:
                    fila[j] = aux.valores[i][j][random.randint(0, len(aux.valores[i][j]) - 1)]

            # Si no es un tablero válido, inténtalo de nuevo.
            while len(list(set(fila))) != NB:
                for j in range(0, NB):
                    if valores[i][j] == 0:
                        fila[j] = aux.valores[i][j][random.randint(0, len(aux.valores[i][j]) - 1)]

            c.valores[i] = fila

        poblacion.append(c)

    # Calcular la aptitud de toda la población.
    poblacion = actualizar_aptitud(poblacion)

    return poblacion


def busqueda_local_columnas(poblacion):
    """
    Realiza una búsqueda local en la población, enfocándose en corregir columnas ilegales en cada individuo.

    Parámetros:
    - poblacion: Lista de individuos (instancias de Poblacion) en la población.

    Retorna:
    - La población con las columnas ilegales corregidas.
    """
    for individuo in poblacion:

        columnas_ilegales = encontrar_columnas_ilegales(individuo.valores)
        for columna in columnas_ilegales:
            duplicados = np.zeros(NB, dtype=bool)
            duplicados2 = np.zeros(NB, dtype=bool)

            columna2 = random.choice(columnas_ilegales)

            _, indices1, counts1 = np.unique(individuo.valores[columna], return_index=True, return_counts=True)
            duplicados[indices1[counts1 > 1]] = True

            _, indices2, counts2 = np.unique(individuo.valores[columna2], return_index=True, return_counts=True)
            duplicados2[indices2[counts2 > 1]] = True
            # Intercambiar elementos si ambos tienen duplicados en la misma posición
            for i in range(NB):
                if duplicados[i] and duplicados[i]:
                    if not columna_duplicada(individuo.valores, columna,
                                             individuo.valores[columna2][i]) and not columna_duplicada(
                            individuo.valores, columna2, individuo.valores[columna][i]):
                        individuo.valores[columna][i], individuo.valores[columna2][i] = individuo.valores[columna2][i], \
                        individuo.valores[columna][i]
                        break
        individuo.funcion_aptitud()
    return poblacion
    # Actualizar la aptitud después de la búsqueda local


def bloque_tiene_repetidos(valores, bloque):
    """
    Determina si un bloque específico en un tablero de Sudoku tiene valores repetidos.

    Parámetros:
    - valores: Matriz 2D que representa el tablero de Sudoku.
    - bloque: Índice del bloque a verificar (0-8).

    Retorna:
    - True si hay valores repetidos en el bloque, False de lo contrario.
    """
    row_start, col_start = divmod(bloque, 3)
    block_values = valores[row_start * 3: (row_start + 1) * 3, col_start * 3: (col_start + 1) * 3].flatten()
    return np.any((block_values != 0) & (np.isin(block_values, block_values[block_values != 0])))


def busqueda_local_subbloques(poblacion):
    """
    Realiza una búsqueda local en la población, enfocándose en corregir subbloques ilegales en cada individuo.

    Parámetros:
    - poblacion: Lista de individuos (instancias de Poblacion) en la población.

    Retorna:
    - La población con los subbloques ilegales corregidos.
    """
    for individuo in poblacion:
        bloques_ilegales = encontrar_bloques_ilegales(individuo.valores)
        for bloque in bloques_ilegales:
            bloque2 = random.choice(bloques_ilegales)

            fila, col = divmod(bloque, 3)
            fila2, col2 = divmod(bloque2, 3)
            # Intercambiar elementos si ambos tienen duplicados en la misma posición
            for i in range(3):
                for j in range(3):

                    if bloque_tiene_duplicados_en_posicion(individuo.valores, bloque, i,
                                                           j) and bloque_tiene_duplicados_en_posicion(individuo.valores,
                                                                                                      bloque2, i, j):
                        if not bloque_duplicado(individuo.valores, fila, col,
                                                individuo.valores[fila2 + i][col2 + j]) and not bloque_duplicado(
                                individuo.valores, fila2, col2, individuo.valores[fila + i][col + j]):
                            individuo.valores[fila + i][col + j], individuo.valores[fila2 + i][col2 + j] = \
                            individuo.valores[fila2 + i][col2 + j], individuo.valores[fila + i][col + j]
                            break

        individuo.funcion_aptitud()

    return poblacion


def bloque_tiene_duplicados_en_posicion(valores, bloque, fila, columna):
    """
    Verifica si un valor en una posición específica de un bloque está duplicado.

    Parámetros:
    - valores: Matriz 2D que representa el tablero de Sudoku.
    - bloque: Índice del bloque a verificar.
    - fila: Fila dentro del bloque (0-2).
    - columna: Columna dentro del bloque (0-2).

    Retorna:
    - True si el valor está duplicado en el bloque, False de lo contrario.
    """
    row_start, col_start = divmod(bloque, 3)
    block_values = valores[row_start * 3: (row_start + 1) * 3, col_start * 3: (col_start + 1) * 3]
    value = block_values[fila, columna]
    return value != 0 and np.count_nonzero(block_values == value) > 1


def encontrar_columnas_ilegales(valores):
    """
    Identifica las columnas ilegales en un tablero de Sudoku.

    Parámetros:
    - valores: Matriz 2D que representa el tablero de Sudoku.

    Retorna:
    - Una lista de índices de columnas ilegales.
    """

    columnas_incorrectas = 0
    columnas = []
    for columna in range(9):
        for valor in range(1,10):
            if numero_columna_duplicadas(valores, columna, valor):
                columnas.append(columna)
                break
    return columnas


def encontrar_bloques_ilegales(valores):
    """
    Identifica los bloques ilegales en un tablero de Sudoku.
    Parámetros:
    - valores: Matriz 2D que representa el tablero de Sudoku.

    Retorna:
    - Una lista de índices de bloques ilegales.
    """
    columnas_incorrectas = 0
    bloques = []
    for fila in range(3):
        for columna in range(3):
            for valor in range(1,10):
                if numero_bloque_duplicados(valores, fila, columna, valor):
                    bloques.append((fila * 3 + columna))
                    break
    return bloques


def cruza_de_filas(fila1, fila2):
    """
    Realiza un cruce entre dos filas de Sudoku, generando dos nuevas filas.
    Parámetros:
    - fila1: La primera fila de Sudoku para el cruce.
    - fila2: La segunda fila de Sudoku para el cruce.

    Retorna:
    - Dos nuevas filas resultantes del cruce.
    """

    fila_hijo1 = np.zeros(NB)
    fila_hijo2 = np.zeros(NB)

    restantes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ciclo = 0

    while (0 in fila_hijo1) and (0 in fila_hijo2):  # Mientras las filas hijas no estén completas
        if ciclo % 2 == 0:
            # Asignar el próximo valor no utilizado.
            indice = valores_restantes(fila1, restantes)
            inicio = fila1[indice]
            if fila1[indice] in restantes:
                restantes.remove(fila1[indice])
            else:
                ciclo += 1
                break
            fila_hijo1[indice] = fila1[indice]
            fila_hijo2[indice] = fila2[indice]
            siguiente = fila2[indice]

            while siguiente != inicio:
                indice = encontrar_valor(fila1, siguiente)
                if indice == -1:
                    break
                fila_hijo1[indice] = fila1[indice]
                if fila1[indice] in restantes:
                    restantes.remove(fila1[indice])
                else:
                    break
                fila_hijo2[indice] = fila2[indice]
                siguiente = fila2[indice]

            ciclo += 1

        else:  # invertir valores.
            indice = valores_restantes(fila1, restantes)
            inicio = fila1[indice]
            if fila1[indice] in restantes:
                restantes.remove(fila1[indice])
            else:
                ciclo += 1
                break
            fila_hijo1[indice] = fila2[indice]
            fila_hijo2[indice] = fila1[indice]
            siguiente = fila2[indice]

            while siguiente != inicio:
                indice = encontrar_valor(fila1, siguiente)
                if indice == -1:
                    break
                fila_hijo1[indice] = fila2[indice]
                if fila1[indice] in restantes:
                    restantes.remove(fila1[indice])
                else:
                    break
                fila_hijo2[indice] = fila1[indice]
                siguiente = fila2[indice]

            ciclo += 1

    return fila_hijo1, fila_hijo2


NB = 9


def resolver_sudoku(Nc, tasa_mutacion, tasa_seleccion, sudoku_board, board_recall, label_recall):
    """
    Resuelve un Sudoku utilizando un algoritmo genético.

    Parámetros:
    - Nc: Número de cromosomas (individuos) en la población.
    - tasa_mutacion: Probabilidad de que ocurra una mutación en un individuo.
    - tasa_seleccion: Probabilidad de seleccionar el mejor individuo en el torneo.
    - sudoku_board: Matriz 1D que representa el tablero de Sudoku a resolver.
    - board_recall: Función de callback para actualizar el tablero visualmente.
    - label_recall: Función de callback para actualizar la etiqueta de generación y aptitud.

    El algoritmo comienza creando una población inicial, luego pasa por un número fijo de generaciones
    (o hasta encontrar una solución). En cada generación, se realiza selección, cruce, mutación y una búsqueda local
    para mejorar la aptitud de los individuos. Los mejores individuos (élites) se preservan entre generaciones.
    Si la población se estanca, se reinicia la población.

    Retorna:
    - La solución del Sudoku como una matriz 2D si se encuentra una solución, de lo contrario None.
    """
    N_elites = int(0.1 * Nc)  # Número de élites.
    generaciones = 10000  # Número de generaciones.
    valores = np.zeros((NB, NB), dtype=int)

    valores = sudoku_board.reshape((NB, NB)).astype(int)
    print(valores)
    poblacion = []

    poblacion = crear_poblacion(Nc, valores)

    estancado = 0
    for generacion in range(0, generaciones):

        print("Generación %d" % generacion)

        # Comprobar si hay una solución
        mejor_aptitud = 0.0
        for c in range(0, Nc):
            aptitud = poblacion[c].aptitud

            if aptitud == 1 and not (np.any(poblacion[c].valores == 0)):
                print("¡Solución encontrada en la generación %d!" % generacion)
                print(poblacion[c].valores)

                # Actualiza la interfaz con la función de recall
                board_recall(poblacion[c].valores.reshape(9, 9))
                return poblacion[c].valores.reshape(9, 9)

            # Mejor individuo
            if aptitud > mejor_aptitud:
                mejor_aptitud = aptitud

        print("Mejor aptitud: %f" % mejor_aptitud)
        # Actualiza la interfaz del progreso con la función de recall
        label_recall(generacion, mejor_aptitud)

        siguiente_poblacion = []

        # Seleccionar élites y preservarlas para la próxima generación.
        poblacion.sort(key=lambda x: x.aptitud, reverse=True)
        elites = []
        for e in range(0, N_elites):
            elite = Poblacion()
            elite.valores = np.copy(poblacion[e].valores)
            elites.append(elite)

        # Crear el resto de la población.
        for _ in range(N_elites, Nc, 2):
            # Seleccionar padres de la población a través de un torneo.

            padre1 = torneo(poblacion, tasa_seleccion)
            padre2 = torneo(poblacion, tasa_seleccion)

            hijo1, hijo2 = cruza(padre1, padre2, tasa_seleccion)

            # Mutar hijo1.
            hijo1.aptitud
            hijo1.mutar(tasa_mutacion, valores)
            hijo1.funcion_aptitud()

            # Mutar hijo2.
            hijo2.aptitud
            hijo2.mutar(tasa_mutacion, valores)
            hijo2.funcion_aptitud()

            # Agregar hijos a la nueva población.
            siguiente_poblacion.append(hijo1)
            siguiente_poblacion.append(hijo2)

        siguiente_poblacion = busqueda_local_columnas(siguiente_poblacion)
        siguiente_poblacion = busqueda_local_subbloques(siguiente_poblacion)

        # Añadir élites al final de la población. Estos no habrán sido afectados por el cruce o la mutación.
        for e in range(0, N_elites):
            siguiente_poblacion.append(elites[e])

        # Seleccionar próxima generación.
        poblacion = siguiente_poblacion
        poblacion = actualizar_aptitud(poblacion)

        # Comprobar si la población está estancada.
        poblacion.sort(key=lambda x: x.aptitud, reverse=True)
        mejor_individuo = poblacion[0]
        peor_individuo = poblacion[-1]
        if poblacion[0].aptitud != poblacion[1].aptitud:
            estancado = 0
        else:
            estancado += 1

        # Re-crear la población si han pasado 100 generaciones con las dos poblaciones más aptas siempre teniendo la misma aptitud.
        if estancado >= 100:
            print("Reiniciando la poblacion")
            poblacion = crear_poblacion(Nc, valores)
            estancado = 0

            # Actualiza la interfaz con la función de recall
            board_recall(poblacion[0].valores.reshape(9, 9))

    print("No se encontró ninguna solución.")
    return None
