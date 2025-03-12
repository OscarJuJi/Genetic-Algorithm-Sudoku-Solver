import threading
import numpy as np
from algoritmo import resolver_sudoku
from tkinter import ttk
import tkinter as tk

sudoku_easy = np.array("0 0 9 0 0 0 1 0 0 2 1 7 0 0 0 3 6 8 0 0 0 2 0 7 0 0 0 0 6 4 1 0 3 5 8 0 0 7 0 0 0 0 0 3 0 1 "
                        "5 0 4 2 8 0 7 9 0 0 0 5 8 9 0 0 0 4 8 5 0 0 0 2 9 3 0 0 6 3 0 2 8 0 0".split()).astype(int)
sudoku_inter = np.array("0 1 0 5 0 6 0 2 0 3 0 0 0 0 0 0 0 6 0 0 9 1 0 4 5 0 0 0 9 0 0 1 0 0 4 0 0 7 0 3 0 2 0 5 0 0 "
                        "3 0 0 8 0 0 6 0 0 0 3 2 0 7 1 0 0 9 0 0 0 0 0 0 0 2 0 5 0 6 0 1 0 8 0".split()).astype(int)
sudoku_hard = np.array("0 3 0 0 7 0 0 5 0 5 0 0 1 0 6 0 0 9 0 0 1 0 0 0 4 0 0 0 9 0 0 5 0 0 6 0 6 0 0 4 0 2 0 0 7 0 4 "
                       "0 0 1 0 0 3 0 0 0 2 0 0 0 8 0 0 9 0 0 3 0 5 0 0 2 0 1 0 0 2 0 0 7 0".split()).astype(int)


class SudokuBoardGUI:
    """
    Interfaz gráfica de usuario para resolver Sudokus utilizando un algoritmo genético.
    """

    def __init__(self, root):
        """
        Inicializa la interfaz gráfica de usuario para el tablero de Sudoku.

        Parámetros:
        - root: Instancia Tkinter que actúa como ventana principal para la GUI.
        """
        self.progress_label = None
        self.sudoku_canvas = None
        self.population_var = None
        self.mutation_var = None
        self.crossovers_var = None
        self.root = root
        self.root.title("Sudoku Board")

        # Variables to store input values
        self.difficulty_var = tk.StringVar()

        # Init empty board
        self.sudoku_board = np.zeros((9, 9), dtype=int)

        # Create and set widgets
        self.create_widgets()

    def create_widgets(self):
        """
        Crea y organiza los widgets (elementos de la interfaz gráfica) en la ventana principal.
        Incluye la configuración de la dificultad, entrada para parámetros del algoritmo y botones de control.
        """
        # Changing root background color
        self.root.configure(bg="white")

        self.difficulty_var = tk.StringVar()
        self.crossovers_var = tk.DoubleVar()
        self.mutation_var = tk.DoubleVar()
        self.population_var = tk.IntVar()

        # Widget frame
        main_frame = ttk.Frame(self.root, style="TFrame")
        main_frame.grid(padx=10, pady=10)

        # Main frame style
        style = ttk.Style()
        style.configure("TFrame", background="white")
        style.configure("TLabel", background="white")

        # Difficulty label
        difficulty_label = ttk.Label(main_frame, text="Difficulty:", style="TLabel")
        difficulty_label.grid(row=0, column=0, padx=10, pady=10)

        # Drop-down menu for difficulty selection
        difficulty_values = ["easy", "intermediate", "hard"]
        difficulty_combobox = ttk.Combobox(main_frame, textvariable=self.difficulty_var, values=difficulty_values)
        difficulty_combobox.grid(row=0, column=1, padx=10, pady=10)

        # Set default value
        difficulty_combobox.set(difficulty_values[0])

        # Difficulty button
        apply_button = ttk.Button(main_frame, text="Apply", command=self.apply_difficulty)
        apply_button.grid(row=0, column=2, padx=10, pady=10)

        # Progress label
        self.progress_label = ttk.Label(main_frame, text="Generation 0\tBest fitness: 0", style="TLabel")
        self.progress_label.grid(row=6, column=0, columnspan=3, padx=10, pady=10)

        # Sudoku canvas
        self.sudoku_canvas = tk.Canvas(main_frame, width=268, height=268, borderwidth=1, relief="solid", bg="white")
        self.sudoku_canvas.grid(row=1, column=0, columnspan=3)

        # Drawing empty board
        self.draw_sudoku_board()

        # Input and labels for parameters
        crossover_label = ttk.Label(main_frame, text="Crossover rate:")
        crossover_label.grid(row=2, column=0, padx=20, pady=10)

        mutation_label = ttk.Label(main_frame, text="Mutation:")
        mutation_label.grid(row=3, column=0, padx=20, pady=10)

        population_label = ttk.Label(main_frame, text="Population size:")
        population_label.grid(row=4, column=0, padx=20, pady=10)

        # Input crossover
        crossover_rate_input = ttk.Entry(main_frame, textvariable=self.crossovers_var, validate="key",
                                         validatecommand=(self.root.register(validate_float), '%P'))
        crossover_rate_input.grid(row=2, column=1, padx=20, pady=10)

        # Input for mutation rate
        mutation_input = ttk.Entry(main_frame, textvariable=self.mutation_var, validate="key",
                                   validatecommand=(self.root.register(validate_float), '%P'))
        mutation_input.grid(row=3, column=1, padx=20, pady=10)

        # Population value input
        population_input = ttk.Entry(main_frame, textvariable=self.population_var, validate="key",
                                     validatecommand=(self.root.register(validate_int), '%P'))
        population_input.grid(row=4, column=1, padx=20, pady=10)

        # Solve button
        solve_button = ttk.Button(main_frame, text="Resolver", command=self.solve_sudoku, style="TButton")
        solve_button.grid(row=5, column=0, columnspan=2, pady=20)
        style = ttk.Style()
        style.configure("TButton", foreground="black", background="#6495ED", padding=(10, 5))

    def draw_sudoku_board(self):
        """
        Dibuja el tablero de Sudoku en la interfaz, representando el estado actual del tablero.
        Muestra valores y errores si los hay.
        """

        cell_size = 30
        bold_line_width = 2
        normal_line_width = 1

        for i in range(10):
            x0 = i * cell_size
            y0 = 0
            x1 = x0
            y1 = 9 * cell_size
            line_width = bold_line_width if i % 3 == 0 else normal_line_width
            self.sudoku_canvas.create_line(x0, y0, x1, y1, width=line_width, fill="#6495ED")

        for j in range(10):
            x0 = 0
            y0 = j * cell_size
            x1 = 9 * cell_size
            y1 = y0
            line_width = bold_line_width if j % 3 == 0 else normal_line_width
            self.sudoku_canvas.create_line(x0, y0, x1, y1, width=line_width, fill="#6495ED")

        # Fill cells with the board values
        for i in range(9):
            for j in range(9):
                value = self.sudoku_board[i, j]
                if value != 0:
                    x = j * cell_size + cell_size / 2
                    y = i * cell_size + cell_size / 2
                    # Changing to red color if is an error
                    color = "red" if self.is_error(i, j) else "black"
                    self.sudoku_canvas.create_text(x, y, text=str(value), font=("Helvetica", 12, "bold"), fill=color)

    def apply_difficulty(self):
        """
        Aplica la dificultad seleccionada al tablero de Sudoku y actualiza la visualización.
        Carga un tablero de Sudoku basado en la dificultad seleccionada (fácil, intermedio, difícil).
        """
        difficulty = self.difficulty_var.get()

        # Assigning the board as selected difficulty
        if difficulty == "easy":
            self.sudoku_board = sudoku_easy.reshape((9, 9))
        elif difficulty == "intermediate":
            self.sudoku_board = sudoku_inter.reshape((9, 9))
        elif difficulty == "hard":
            self.sudoku_board = sudoku_hard.reshape((9, 9))

        # Clearing the canvas before drawing a new one
        self.sudoku_canvas.delete("all")

        # Drawing the new board
        self.draw_sudoku_board()

    def solve_sudoku(self):
        """
        Inicia la resolución del Sudoku utilizando un hilo separado.
        Esto evita que la interfaz gráfica se congele durante la ejecución del algoritmo genético.
        """
        # Starting new process on a new thread
        threading.Thread(target=self._solve_parallel_sudoku, daemon=True).start()

    def _solve_parallel_sudoku(self):
        """
        Método que se ejecuta en un hilo paralelo para resolver el Sudoku.
        Obtiene los parámetros de entrada de la interfaz y llama a la función de resolución.
        """
        # Getting input values
        crossovers = self.crossovers_var.get()
        mutations = self.mutation_var.get()
        population = self.population_var.get()

        # Calling the function sudoku_solver with the parameters
        resolver_sudoku(population, mutations, crossovers, self.sudoku_board, self.update_board_parallel,
                        self.update_progress_parallel)

    def update_board_parallel(self, board):
        """
        Método paralelo para actualizar el tablero de Sudoku en la interfaz.
        Se invoca desde un hilo paralelo y usa 'after' para realizar cambios en la interfaz.
        """

        # Parallel method for updating interface from secondary thread
        self.sudoku_canvas.after(0, self.update_board, board)

    def update_board(self, board):
        """
        Actualiza el tablero de Sudoku en la interfaz con una nueva solución o estado.
        Se llama para reflejar los cambios en el tablero después de cada generación del algoritmo.
        """

        # Updating the board with the new solution
        self.sudoku_board = board
        # Clearing the canvas before drawing the new board
        self.sudoku_canvas.delete("all")
        # Draw updated board
        self.draw_sudoku_board()

    def update_progress_parallel(self, generation, best_fitness):
        """
        Método paralelo para actualizar la información de progreso en la interfaz.
        Muestra la generación actual y la mejor aptitud encontrada hasta el momento.
        """
        self.root.after(0, self.update_progress, generation, best_fitness)

    def update_progress(self, generation, best_fitness):
        """
        Actualiza la información de progreso en la interfaz.
        Se llama para reflejar el progreso del algoritmo genético en la interfaz.
        """
        self.progress_label.config(text=f"Generation {generation}\tBest fitness: {best_fitness:.2f}")

    def is_error(self, i, j):
        """
        Verifica si hay un conflicto con el número en la celda (i, j) del tablero de Sudoku.
        Comprueba si el mismo número aparece en la misma fila, columna o bloque 3x3.

        Parámetros:
        - i: Índice de fila de la celda.
        - j: Índice de columna de la celda.

        Retorna:
        - True si hay un conflicto, False en caso contrario.
        """
        # Verify if the number on the cell (i, j) is in conflict
        number = self.sudoku_board[i, j]

        # Verify row and column
        for k in range(9):
            if k != j and self.sudoku_board[i, k] == number:
                return True
            if k != i and self.sudoku_board[k, j] == number:
                return True

        # Verify 3x3 block
        start_row, start_column = 3 * (i // 3), 3 * (j // 3)
        for fila in range(start_row, start_row + 3):
            for column in range(start_column, start_column + 3):
                if fila != i or column != j:
                    if self.sudoku_board[fila][column] == number:
                        return True

        return False

    def run(self):
        """
        Inicia el bucle principal de la interfaz gráfica.
        Mantiene abierta la ventana y procesa los eventos de la interfaz.
        """
        self.root.mainloop()


def validate_float(new_value):
    """
    Valida si una cadena de caracteres representa un valor flotante válido dentro de un rango específico.

    Esta función se utiliza para validar las entradas de los usuarios en la interfaz gráfica, asegurándose de que
    los valores flotantes estén dentro del rango [0.0, 1.0] y sean adecuados para los parámetros del algoritmo genético.

    Parámetros:
    - new_value: La cadena de caracteres que se va a validar.

    Retorna:
    - True si 'new_value' es un valor flotante válido dentro del rango [0.0, 1.0], o si está vacía.
    - False si 'new_value' no es un valor flotante válido o está fuera del rango.
    """
    try:
        if new_value in ("", "."):
            return True
        float_value = float(new_value)
        return 0.0 <= float_value <= 1.0
    except ValueError:
        return False


def validate_int(new_value):
    """
    Valida si una cadena de caracteres representa un valor entero positivo.

    Esta función se utiliza para validar las entradas de los usuarios en la interfaz gráfica, asegurándose de que
    los valores enteros sean positivos, lo cual es necesario para parámetros como el tamaño de la población.

    Parámetros:
    - new_value: La cadena de caracteres que se va a validar.

    Retorna:
    - True si 'new_value' es una representación válida de un número entero positivo o está vacía.
    - False en caso contrario.
    """

    return new_value.isdigit() or new_value == ""
